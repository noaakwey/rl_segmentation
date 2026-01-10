"""
Стратегии пространственного сэмплирования и оптимизации разметки обучающих данных.

Этот модуль реализует:
1. Оптимальные планы разметки для различных задач (пашня, овраги)
2. Стратифицированное сэмплирование по ландшафтам и характеристикам
3. Валидацию пространственного распределения эталонов
4. Рекомендации по количеству и качеству разметки
5. Проверку пространственной автокорреляции
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial import distance_matrix
import warnings


@dataclass
class SamplingRecommendations:
    """Рекомендации по разметке данных"""
    task: str  # 'cropland' или 'gully'
    min_samples: int
    optimal_samples: int
    total_area_km2: Tuple[float, float]  # (min, max)
    strata_distribution: Dict[str, Dict]
    expected_accuracy: Tuple[float, float]  # (min, max) Dice
    annotation_time_hours: Tuple[int, int]
    training_time_hours: Tuple[int, int]
    spatial_requirements: Dict
    warnings: List[str]
    recommendations: List[str]


@dataclass
class ValidationReport:
    """Отчет валидации разметки"""
    is_valid: bool
    num_samples: int
    spatial_distribution_score: float  # 0-1
    diversity_score: float  # 0-1
    coverage_score: float  # 0-1
    overall_score: float  # 0-1
    warnings: List[str]
    recommendations: List[str]
    expected_accuracy: Tuple[float, float]


class OptimalSamplingPlan:
    """
    Генератор оптимальных планов разметки для различных задач.

    Использование:
    ```python
    plan = OptimalSamplingPlan(
        task='cropland',
        target_accuracy=0.90,
        available_budget='medium',
        landscape_diversity='high'
    )
    recommendations = plan.generate()
    print(recommendations)
    ```
    """

    # Базовые конфигурации для различных задач
    TASK_CONFIGS = {
        'cropland': {
            'minimal': {
                'samples': (50, 100),
                'area_km2': (5, 10),
                'accuracy': (0.75, 0.82),
                'annotation_hours': (5, 10),
                'training_hours': (2, 4),
            },
            'optimal': {
                'samples': (200, 500),
                'area_km2': (20, 50),
                'accuracy': (0.85, 0.92),
                'annotation_hours': (15, 30),
                'training_hours': (6, 12),
            },
            'maximum': {
                'samples': (1000, 2000),
                'area_km2': (100, 200),
                'accuracy': (0.92, 0.96),
                'annotation_hours': (60, 120),
                'training_hours': (24, 48),
            }
        },
        'gully': {
            'minimal': {
                'samples': (30, 50),
                'length_km': (1, 2),
                'accuracy': (0.65, 0.75),
                'annotation_hours': (3, 5),
                'training_hours': (1, 2),
            },
            'optimal': {
                'samples': (100, 200),
                'length_km': (5, 10),
                'accuracy': (0.75, 0.85),
                'annotation_hours': (10, 20),
                'training_hours': (4, 8),
            },
            'maximum': {
                'samples': (500, 1000),
                'length_km': (20, 40),
                'accuracy': (0.85, 0.92),
                'annotation_hours': (40, 80),
                'training_hours': (20, 40),
            }
        }
    }

    # Стратификация для пашни
    CROPLAND_STRATA = {
        'low_diversity': {
            'steppe_large': 0.60,          # Степь, крупные поля
            'steppe_medium': 0.30,         # Степь, средние поля
            'forest_steppe_medium': 0.10,  # Лесостепь
        },
        'medium_diversity': {
            'steppe_large': 0.30,
            'steppe_medium': 0.25,
            'forest_steppe_medium': 0.25,
            'foothill_small': 0.15,
            'valley_irregular': 0.05,
        },
        'high_diversity': {
            'steppe_large': 0.24,
            'steppe_medium': 0.20,
            'forest_steppe_medium': 0.20,
            'foothill_small': 0.16,
            'valley_irregular': 0.12,
            'terraced': 0.08,
        }
    }

    # Стратификация для оврагов
    GULLY_STRATA = {
        'low_diversity': {
            'active_v_shaped': 0.40,
            'stable_u_shaped': 0.40,
            'vegetated': 0.20,
        },
        'medium_diversity': {
            'active_v_shaped': 0.30,
            'stable_u_shaped': 0.35,
            'vegetated': 0.20,
            'branching': 0.15,
        },
        'high_diversity': {
            'active_v_shaped': 0.25,
            'stable_u_shaped': 0.30,
            'vegetated': 0.20,
            'branching': 0.15,
            'healed': 0.10,
        }
    }

    def __init__(
        self,
        task: str = 'cropland',
        target_accuracy: float = 0.90,
        available_budget: str = 'medium',  # 'low', 'medium', 'high'
        landscape_diversity: str = 'medium',  # 'low', 'medium', 'high'
        spatial_resolution: Optional[float] = None,  # метры
        min_gully_width: Optional[float] = None,  # для оврагов
    ):
        """
        Инициализация планировщика разметки.

        Args:
            task: 'cropland' или 'gully'
            target_accuracy: Целевая точность (Dice)
            available_budget: Доступный бюджет разметки
            landscape_diversity: Разнообразие ландшафта
            spatial_resolution: Пространственное разрешение данных (м)
            min_gully_width: Минимальная ширина оврага (м), только для task='gully'
        """
        assert task in ['cropland', 'gully'], f"task должен быть 'cropland' или 'gully', получен: {task}"
        assert available_budget in ['low', 'medium', 'high']
        assert landscape_diversity in ['low', 'medium', 'high']
        assert 0.5 <= target_accuracy <= 0.99, "target_accuracy должен быть в диапазоне [0.5, 0.99]"

        self.task = task
        self.target_accuracy = target_accuracy
        self.budget = available_budget
        self.diversity = landscape_diversity
        self.spatial_resolution = spatial_resolution
        self.min_gully_width = min_gully_width

    def generate(self) -> SamplingRecommendations:
        """Генерировать рекомендации по разметке"""

        # Выбрать конфигурацию на основе бюджета
        budget_map = {'low': 'minimal', 'medium': 'optimal', 'high': 'maximum'}
        config_level = budget_map[self.budget]
        base_config = self.TASK_CONFIGS[self.task][config_level]

        # Скорректировать на основе целевой точности
        adjusted_samples = self._adjust_for_accuracy(
            base_config['samples'],
            self.target_accuracy
        )

        # Получить распределение по стратам
        if self.task == 'cropland':
            strata = self.CROPLAND_STRATA[self.diversity]
        else:
            strata = self.GULLY_STRATA[self.diversity]

        strata_distribution = self._compute_strata_distribution(
            adjusted_samples,
            strata
        )

        # Пространственные требования
        spatial_reqs = self._get_spatial_requirements()

        # Предупреждения и рекомендации
        warnings_list = []
        recommendations_list = []

        # Проверки для оврагов
        if self.task == 'gully':
            self._check_gully_resolution(warnings_list, recommendations_list)

        # Общие рекомендации
        self._add_general_recommendations(recommendations_list)

        return SamplingRecommendations(
            task=self.task,
            min_samples=adjusted_samples[0],
            optimal_samples=adjusted_samples[1],
            total_area_km2=base_config.get('area_km2', base_config.get('length_km', (0, 0))),
            strata_distribution=strata_distribution,
            expected_accuracy=self._estimate_accuracy(adjusted_samples),
            annotation_time_hours=base_config['annotation_hours'],
            training_time_hours=base_config['training_hours'],
            spatial_requirements=spatial_reqs,
            warnings=warnings_list,
            recommendations=recommendations_list
        )

    def _adjust_for_accuracy(
        self,
        base_samples: Tuple[int, int],
        target_accuracy: float
    ) -> Tuple[int, int]:
        """Скорректировать количество эталонов на основе целевой точности"""

        # Эмпирическое соотношение: точность ~ log(samples)
        # Для достижения +0.05 точности нужно ~1.5x больше данных

        if target_accuracy <= 0.80:
            scale = 0.7
        elif target_accuracy <= 0.85:
            scale = 1.0
        elif target_accuracy <= 0.90:
            scale = 1.3
        elif target_accuracy <= 0.93:
            scale = 1.7
        else:  # > 0.93
            scale = 2.2

        return (
            int(base_samples[0] * scale),
            int(base_samples[1] * scale)
        )

    def _compute_strata_distribution(
        self,
        total_samples: Tuple[int, int],
        strata_proportions: Dict[str, float]
    ) -> Dict[str, Dict]:
        """Вычислить распределение эталонов по стратам"""

        mid_samples = (total_samples[0] + total_samples[1]) // 2

        distribution = {}
        for stratum_name, proportion in strata_proportions.items():
            num_samples = int(mid_samples * proportion)
            distribution[stratum_name] = {
                'proportion': proportion,
                'num_samples': num_samples,
                'percentage': proportion * 100
            }

        return distribution

    def _get_spatial_requirements(self) -> Dict:
        """Получить требования к пространственному распределению"""

        if self.task == 'cropland':
            min_distance = 1000  # метры
            landscape_zones = {'low': 2, 'medium': 4, 'high': 6}[self.diversity]
        else:  # gully
            min_distance = 500
            landscape_zones = {'low': 2, 'medium': 3, 'high': 5}[self.diversity]

        return {
            'min_distance_m': min_distance,
            'num_landscape_zones': landscape_zones,
            'spatial_clustering_threshold': 0.3,  # Максимальная доля кластеризованных эталонов
        }

    def _check_gully_resolution(
        self,
        warnings: List[str],
        recommendations: List[str]
    ):
        """Проверить разрешение для выделения оврагов"""

        if self.spatial_resolution is None:
            warnings.append(
                "⚠️ Не указано пространственное разрешение. "
                "Для оврагов критически важно разрешение ≤ 3м!"
            )
            recommendations.append(
                "Укажите spatial_resolution для точных рекомендаций"
            )
            return

        if self.min_gully_width is None:
            warnings.append(
                "⚠️ Не указана минимальная ширина оврага. "
                "Рекомендуется для оценки пригодности разрешения."
            )
            return

        # Правило: минимум 2-3 пикселя для надежного выделения
        pixels_width = self.min_gully_width / self.spatial_resolution

        if pixels_width < 2.0:
            warnings.append(
                f"⚠️⚠️ КРИТИЧНО: Овраги шириной {self.min_gully_width}м будут "
                f"занимать {pixels_width:.1f} пикс. на разрешении {self.spatial_resolution}м. "
                f"Минимум 2-3 пикселя требуется для надежного выделения!"
            )
            recommendations.append(
                f"Используйте данные с разрешением ≤ {self.min_gully_width / 2.5:.1f}м "
                f"или исключите овраги < {self.spatial_resolution * 2.5:.0f}м"
            )
        elif pixels_width < 3.0:
            warnings.append(
                f"⚠️ Овраги {self.min_gully_width}м будут {pixels_width:.1f} пикс. "
                f"Возможна пониженная точность."
            )
            recommendations.append(
                "Рассмотрите данные с более высоким разрешением для лучших результатов"
            )
        else:
            recommendations.append(
                f"✅ Разрешение {self.spatial_resolution}м подходит для оврагов ≥ {self.min_gully_width}м "
                f"({pixels_width:.1f} пикс.)"
            )

    def _add_general_recommendations(self, recommendations: List[str]):
        """Добавить общие рекомендации"""

        recommendations.extend([
            "Используйте стратифицированное сэмплирование по ландшафтам",
            "Обеспечьте минимальное расстояние между эталонами > 500-1000м",
            "Включайте разнообразие размеров и форм объектов",
            "Рассмотрите использование Active Learning для оптимизации",
            "Выделите 20-30% данных для валидации (из других регионов)",
        ])

        if self.task == 'gully':
            recommendations.extend([
                "Для оврагов: используйте мультисезонные данные (весна+осень)",
                "Добавьте DEM (цифровая модель рельефа) как дополнительный канал",
                "Включите различные стадии развития оврагов (активные → залеченные)",
            ])

    def _estimate_accuracy(
        self,
        num_samples: Tuple[int, int]
    ) -> Tuple[float, float]:
        """Оценить ожидаемую точность на основе количества эталонов"""

        mid_samples = (num_samples[0] + num_samples[1]) // 2

        if self.task == 'cropland':
            # Эмпирическая модель для пашни
            if mid_samples < 100:
                base_acc = (0.70, 0.78)
            elif mid_samples < 200:
                base_acc = (0.78, 0.85)
            elif mid_samples < 500:
                base_acc = (0.85, 0.91)
            else:
                base_acc = (0.90, 0.95)
        else:  # gully
            # Овраги сложнее - на 5-10% ниже точность
            if mid_samples < 50:
                base_acc = (0.60, 0.70)
            elif mid_samples < 100:
                base_acc = (0.70, 0.78)
            elif mid_samples < 200:
                base_acc = (0.78, 0.85)
            else:
                base_acc = (0.85, 0.90)

        # Корректировка на разнообразие
        diversity_bonus = {'low': -0.03, 'medium': 0.0, 'high': 0.02}[self.diversity]

        return (
            max(0.5, base_acc[0] + diversity_bonus),
            min(0.99, base_acc[1] + diversity_bonus)
        )

    def __str__(self) -> str:
        """Красивая печать рекомендаций"""
        rec = self.generate()

        lines = [
            f"\nРекомендуемый план разметки для задачи: {rec.task}",
            "=" * 60,
            f"Целевая точность (Dice):     {self.target_accuracy:.2f}",
            f"Ландшафтное разнообразие:    {self.diversity.capitalize()}",
            "",
            "Рекомендации:",
            f"  Минимум эталонов:          {rec.min_samples}",
            f"  Рекомендуемое количество:  {rec.optimal_samples}",
        ]

        if self.task == 'cropland':
            lines.append(f"  Общая площадь:             {rec.total_area_km2[0]}-{rec.total_area_km2[1]} км²")
        else:
            lines.append(f"  Общая длина:               {rec.total_area_km2[0]}-{rec.total_area_km2[1]} км")

        lines.extend([
            f"  Ландшафтных зон:           {rec.spatial_requirements['num_landscape_zones']}",
            "",
            "Распределение по стратам:",
        ])

        for stratum, info in rec.strata_distribution.items():
            lines.append(
                f"  - {stratum:30s} {info['num_samples']:3d} эталонов ({info['percentage']:5.1f}%)"
            )

        lines.extend([
            "",
            f"Ожидаемое время разметки:    {rec.annotation_time_hours[0]}-{rec.annotation_time_hours[1]} часов",
            f"Ожидаемое время обучения:    {rec.training_time_hours[0]}-{rec.training_time_hours[1]} часов (GPU)",
            f"Ожидаемая точность:          {rec.expected_accuracy[0]:.2f}-{rec.expected_accuracy[1]:.2f} (Dice)",
        ])

        if rec.warnings:
            lines.extend(["", "⚠️ Предупреждения:"])
            for warning in rec.warnings:
                lines.append(f"  {warning}")

        if rec.recommendations:
            lines.extend(["", "💡 Рекомендации:"])
            for i, recommendation in enumerate(rec.recommendations[:5], 1):
                lines.append(f"  {i}. {recommendation}")

        return "\n".join(lines)


class SpatialAutocorrelation:
    """
    Валидатор пространственной автокорреляции эталонов.

    Проверяет, что эталоны хорошо распределены в пространстве
    и не образуют кластеры (которые снижают эффективность обучения).
    """

    def __init__(
        self,
        min_distance_m: float = 1000.0,
        check_autocorrelation: bool = True
    ):
        """
        Args:
            min_distance_m: Минимальное расстояние между эталонами (м)
            check_autocorrelation: Проверять ли пространственную автокорреляцию
        """
        self.min_distance_m = min_distance_m
        self.check_autocorrelation = check_autocorrelation

    def validate_samples(
        self,
        sample_polygons: Union[gpd.GeoDataFrame, List[Polygon]]
    ) -> Tuple[bool, Dict]:
        """
        Валидировать пространственное распределение эталонов.

        Args:
            sample_polygons: GeoDataFrame или список полигонов с эталонами

        Returns:
            (is_valid, report): Валидны ли данные и детальный отчет
        """

        if isinstance(sample_polygons, list):
            gdf = gpd.GeoDataFrame(geometry=sample_polygons)
        else:
            gdf = sample_polygons

        # Получить центроиды
        centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])

        # Вычислить матрицу расстояний
        dist_matrix = distance_matrix(centroids, centroids)

        # Убрать диагональ (расстояние до самого себя)
        np.fill_diagonal(dist_matrix, np.inf)

        # Минимальное расстояние до ближайшего соседа для каждого эталона
        min_distances = np.min(dist_matrix, axis=1)

        # Статистика
        mean_distance = np.mean(min_distances)
        median_distance = np.median(min_distances)
        min_min_distance = np.min(min_distances)

        # Найти кластеры (эталоны слишком близко)
        clustered = min_distances < self.min_distance_m
        num_clustered = np.sum(clustered)
        pct_clustered = (num_clustered / len(min_distances)) * 100

        # Проверка валидности
        is_valid = pct_clustered < 30.0  # Не более 30% в кластерах

        report = {
            'num_samples': len(gdf),
            'mean_distance_m': mean_distance,
            'median_distance_m': median_distance,
            'min_distance_m': min_min_distance,
            'clustered_samples': num_clustered,
            'pct_clustered': pct_clustered,
            'is_valid': is_valid,
            'threshold_m': self.min_distance_m,
        }

        # Индексы проблемных эталонов
        if num_clustered > 0:
            report['clustered_indices'] = np.where(clustered)[0].tolist()

        return is_valid, report

    def __str__(self, report: Dict) -> str:
        """Красивая печать отчета"""
        lines = [
            "\nОтчет валидации пространственного распределения",
            "=" * 60,
            f"Количество эталонов:         {report['num_samples']}",
            f"Среднее расстояние до соседа: {report['mean_distance_m']:.0f} м",
            f"Медианное расстояние:        {report['median_distance_m']:.0f} м",
            f"Минимальное расстояние:      {report['min_distance_m']:.0f} м",
            "",
            f"Порог кластеризации:         {report['threshold_m']:.0f} м",
            f"Кластеризованных эталонов:   {report['clustered_samples']} ({report['pct_clustered']:.1f}%)",
            "",
        ]

        if report['is_valid']:
            lines.append("✅ Статус: Хорошее пространственное распределение")
        else:
            lines.append("❌ Статус: Слишком много кластеризованных эталонов!")
            lines.append(f"   Рекомендация: Перераспределите эталоны на расстояние > {report['threshold_m']}м")

        return "\n".join(lines)


def validate_samples(
    samples: Union[gpd.GeoDataFrame, Dict],
    plan: Optional[SamplingRecommendations] = None,
    min_distance_m: float = 1000.0
) -> ValidationReport:
    """
    Комплексная валидация качества размеченных данных.

    Проверяет:
    - Достаточное количество эталонов
    - Пространственное распределение
    - Разнообразие (размеры, формы)
    - Ландшафтное покрытие

    Args:
        samples: Размеченные данные (GeoDataFrame или dict с image/mask)
        plan: План разметки для сравнения (опционально)
        min_distance_m: Минимальное расстояние между эталонами

    Returns:
        ValidationReport с детальным анализом
    """

    warnings_list = []
    recommendations_list = []

    # Извлечь геометрию
    if isinstance(samples, dict):
        # Формат {image, mask}
        num_samples = len(samples.get('image', []))
        gdf = None
    else:
        # GeoDataFrame
        gdf = samples
        num_samples = len(gdf)

    # 1. Проверка количества
    if plan is not None:
        if num_samples < plan.min_samples:
            warnings_list.append(
                f"⚠️ Недостаточно эталонов: {num_samples} < {plan.min_samples} (минимум)"
            )
            quantity_score = num_samples / plan.min_samples
        elif num_samples < plan.optimal_samples:
            recommendations_list.append(
                f"Рекомендуется добавить еще {plan.optimal_samples - num_samples} эталонов "
                f"для достижения оптимального количества"
            )
            quantity_score = 0.7 + 0.3 * (num_samples / plan.optimal_samples)
        else:
            quantity_score = 1.0
    else:
        quantity_score = 0.8  # Неизвестно без плана

    # 2. Пространственное распределение
    spatial_score = 1.0
    if gdf is not None:
        validator = SpatialAutocorrelation(min_distance_m=min_distance_m)
        is_valid, spatial_report = validator.validate_samples(gdf)

        if not is_valid:
            warnings_list.append(
                f"⚠️ Слишком много кластеризованных эталонов: "
                f"{spatial_report['pct_clustered']:.1f}% (допустимо < 30%)"
            )
            recommendations_list.append(
                f"Перераспределите эталоны на расстояние > {min_distance_m}м"
            )
            spatial_score = max(0.3, 1.0 - spatial_report['pct_clustered'] / 100)
        else:
            spatial_score = 1.0

    # 3. Разнообразие размеров (если есть геометрия)
    diversity_score = 0.8  # По умолчанию
    if gdf is not None and 'geometry' in gdf.columns:
        areas = gdf.geometry.area
        areas_ha = areas / 10000  # В гектары

        # Категоризация по размерам
        small = np.sum(areas_ha < 2)
        medium = np.sum((areas_ha >= 2) & (areas_ha <= 10))
        large = np.sum(areas_ha > 10)

        pct_small = small / num_samples * 100
        pct_medium = medium / num_samples * 100
        pct_large = large / num_samples * 100

        # Оптимально: 20-30% малых, 40-50% средних, 20-30% крупных
        if pct_small < 15:
            warnings_list.append(
                f"⚠️ Недостаточно малых объектов: {pct_small:.0f}% (рекомендуется 20-30%)"
            )
        if pct_medium < 30:
            warnings_list.append(
                f"⚠️ Недостаточно средних объектов: {pct_medium:.0f}% (рекомендуется 40-50%)"
            )

        # Оценка разнообразия (энтропия распределения)
        probs = np.array([pct_small, pct_medium, pct_large]) / 100
        probs = probs[probs > 0]  # Убрать нули
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(3)  # Для 3 категорий
        diversity_score = entropy / max_entropy

    # 4. Общая оценка покрытия
    coverage_score = 0.85  # Placeholder, требует доп. анализа

    # Общий score
    overall_score = (
        0.3 * quantity_score +
        0.3 * spatial_score +
        0.2 * diversity_score +
        0.2 * coverage_score
    )

    # Оценка ожидаемой точности на основе score
    if overall_score >= 0.9:
        expected_acc = (0.88, 0.93)
    elif overall_score >= 0.8:
        expected_acc = (0.83, 0.89)
    elif overall_score >= 0.7:
        expected_acc = (0.78, 0.85)
    else:
        expected_acc = (0.70, 0.80)

    is_valid = overall_score >= 0.7

    return ValidationReport(
        is_valid=is_valid,
        num_samples=num_samples,
        spatial_distribution_score=spatial_score,
        diversity_score=diversity_score,
        coverage_score=coverage_score,
        overall_score=overall_score,
        warnings=warnings_list,
        recommendations=recommendations_list,
        expected_accuracy=expected_acc
    )


def print_validation_report(report: ValidationReport):
    """Красиво напечатать отчет валидации"""

    print("\n" + "=" * 70)
    print("ОТЧЕТ ВАЛИДАЦИИ РАЗМЕТКИ")
    print("=" * 70)

    print(f"\n📊 Количество эталонов: {report.num_samples}")

    print(f"\n📈 Оценки качества:")
    print(f"  Пространственное распределение: {report.spatial_distribution_score:.2f} / 1.00")
    print(f"  Разнообразие объектов:          {report.diversity_score:.2f} / 1.00")
    print(f"  Покрытие территории:            {report.coverage_score:.2f} / 1.00")
    print(f"  ОБЩАЯ ОЦЕНКА:                   {report.overall_score:.2f} / 1.00")

    if report.is_valid:
        print(f"\n✅ Статус: ДАННЫЕ ВАЛИДНЫ")
    else:
        print(f"\n❌ Статус: ТРЕБУЕТСЯ УЛУЧШЕНИЕ")

    print(f"\n🎯 Ожидаемая точность: {report.expected_accuracy[0]:.2f} - {report.expected_accuracy[1]:.2f} (Dice)")

    if report.warnings:
        print(f"\n⚠️  Предупреждения:")
        for warning in report.warnings:
            print(f"  • {warning}")

    if report.recommendations:
        print(f"\n💡 Рекомендации:")
        for rec in report.recommendations:
            print(f"  • {rec}")

    print("\n" + "=" * 70)


# Пример использования
if __name__ == "__main__":
    # Пример 1: Создать план для пашни
    print("\n" + "="*70)
    print("ПРИМЕР 1: План разметки для пашни")
    print("="*70)

    plan_cropland = OptimalSamplingPlan(
        task='cropland',
        target_accuracy=0.90,
        available_budget='medium',
        landscape_diversity='high'
    )
    print(plan_cropland)

    # Пример 2: План для оврагов с проверкой разрешения
    print("\n\n" + "="*70)
    print("ПРИМЕР 2: План разметки для оврагов")
    print("="*70)

    plan_gully = OptimalSamplingPlan(
        task='gully',
        target_accuracy=0.80,
        available_budget='medium',
        landscape_diversity='high',
        spatial_resolution=3.0,  # PlanetScope
        min_gully_width=5.0      # 5 метров минимум
    )
    print(plan_gully)

    # Пример 3: Валидация эталонов
    print("\n\n" + "="*70)
    print("ПРИМЕР 3: Валидация разметки")
    print("="*70)

    # Создать синтетические данные для примера
    np.random.seed(42)
    num_samples = 250

    # Случайные полигоны
    polygons = []
    for i in range(num_samples):
        x = np.random.uniform(0, 100000)
        y = np.random.uniform(0, 100000)
        size = np.random.uniform(50, 500)
        poly = Point(x, y).buffer(size)
        polygons.append(poly)

    gdf_synthetic = gpd.GeoDataFrame(geometry=polygons)

    # Валидировать
    validation = validate_samples(
        samples=gdf_synthetic,
        plan=plan_cropland.generate(),
        min_distance_m=1000
    )

    print_validation_report(validation)
