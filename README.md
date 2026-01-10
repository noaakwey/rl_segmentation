# RL-based Crop Segmentation Pipeline

Полный пайплайн для обучения с подкреплением (Reinforcement Learning) для решения задачи сегментации пашни по спутниковым снимкам.

## 📋 Описание

Этот проект реализует инновационный подход к сегментации сельскохозяйственных полей, используя методы обучения с подкреплением. Агент учится сегментировать пашню, получая вознаграждения за правильную сегментацию на основе ground truth данных из shapefile.

### Ключевые особенности

- 🛰️ **Поддержка геопространственных данных**: Работа с GeoTIFF и Shapefile
- 🧠 **RL агенты**: Реализованы PPO (Proximal Policy Optimization) и DQN
- 📊 **Гибкая система вознаграждений**: Dice, IoU, Pixel Accuracy и комбинированные метрики
- 🎯 **Два режима действий**: Pixel-wise и Patch-based сегментация
- 📈 **Мониторинг обучения**: Интеграция с Weights & Biases
- 🔧 **Модульная архитектура**: Легко расширяемый и настраиваемый код

## 🚀 Быстрый старт

### Установка

```bash
# Клонировать репозиторий
git clone <repository-url>
cd rl_segmentation

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt
```

### Подготовка данных

Поместите ваши данные в директорию `data/raw/`:

```
data/raw/
├── satellite_image.tif      # GeoTIFF спутниковый снимок
└── crop_boundaries.shp      # Shapefile с границами пашни
```

**Поддерживаемые форматы:**
- **Изображения**: GeoTIFF (RGB, мультиспектральные, ортофотопланы)
- **Маски**: Shapefile (.shp) с полигонами границ пашни

### Сколько эталонов нужно?

**Для пашни:**
- **Минимум:** 50-100 полей (5-10 км²) → точность ~0.75-0.82
- **Оптимум:** 200-500 полей (20-50 км²) → точность ~0.85-0.92
- **С Active Learning:** старт с 100 → итеративно до 200-300

**Для оврагов (permanent gully):**
- **Минимум:** 30-50 оврагов (1-2 км линейной длины) → точность ~0.65-0.75
- **Оптимум:** 100-200 оврагов (5-10 км) → точность ~0.75-0.85
- ⚠️ **Требуется разрешение ≤ 3м** (PlanetScope, aerial/drone)

**Пространственное распределение:**
- ✅ Минимум 500-1000м между эталонами
- ✅ Стратифицировать по 3-5 ландшафтным зонам
- ✅ Разнообразие размеров и форм
- ❌ Избегать кластеризации в одном регионе

**Автоматические рекомендации:**
```python
from src.sampling_strategies import OptimalSamplingPlan

# Получить план разметки
plan = OptimalSamplingPlan(
    task='cropland',  # или 'gully'
    target_accuracy=0.90,
    available_budget='medium',
    landscape_diversity='high'
)
print(plan)  # Детальные рекомендации
```

📖 **Подробное руководство:** [TRAINING_SAMPLES_GUIDE.md](TRAINING_SAMPLES_GUIDE.md)

### Обучение модели

#### Базовое обучение

```bash
python src/train.py \
    --image data/raw/satellite_image.tif \
    --shapefile data/raw/crop_boundaries.shp \
    --output experiments/my_experiment
```

#### Обучение с конфигурационным файлом

```bash
# Быстрое обучение для экспериментов
python src/train.py --config configs/fast_training.yaml

# Высокое качество для production
python src/train.py --config configs/high_quality.yaml

# Стандартная конфигурация
python src/train.py --config configs/default_config.yaml
```

### Оценка модели

```bash
python src/evaluate.py \
    --checkpoint experiments/my_experiment/checkpoints/best_model.pt \
    --config configs/default_config.yaml \
    --output evaluation_results \
    --n_samples 10 \
    --save_full
```

## 📁 Структура проекта

```
rl_segmentation/
├── src/
│   ├── __init__.py              # Инициализация пакета
│   ├── data_loader.py           # Загрузка GeoTIFF и SHP
│   ├── environment.py           # RL среда для сегментации
│   ├── agent.py                 # PPO и DQN агенты
│   ├── models.py                # Нейросетевые архитектуры
│   ├── rewards.py               # Функции вознаграждения
│   ├── train.py                 # Скрипт обучения
│   └── evaluate.py              # Скрипт оценки
├── utils/
│   ├── __init__.py
│   ├── visualization.py         # Визуализация результатов
│   └── helpers.py               # Вспомогательные функции
├── configs/
│   ├── default_config.yaml      # Стандартная конфигурация
│   ├── fast_training.yaml       # Быстрое обучение
│   └── high_quality.yaml        # Высокое качество
├── data/
│   ├── raw/                     # Исходные данные
│   └── processed/               # Обработанные данные
├── experiments/                 # Результаты экспериментов
├── notebooks/                   # Jupyter notebooks
├── requirements.txt             # Зависимости
└── README.md                    # Документация
```

## 🎓 Как это работает

### 1. Загрузка данных

`GeoDataLoader` загружает спутниковые снимки (GeoTIFF) и векторные данные границ (SHP), конвертирует их в растровую маску и разбивает на патчи для обучения.

```python
from src.data_loader import GeoDataLoader

loader = GeoDataLoader(
    image_path="data/raw/satellite_image.tif",
    shapefile_path="data/raw/crop_boundaries.shp",
    patch_size=256,
    stride=128
)

image, mask = loader.load_all()
patches = loader.extract_patches()
```

### 2. RL среда

`CropSegmentationEnv` - Gymnasium-совместимая среда, где агент учится сегментировать изображение:

- **Наблюдение**: Текущее изображение + текущая предсказанная маска
- **Действие**: Классификация пикселя/патча (пашня/не пашня)
- **Вознаграждение**: Метрики качества сегментации (Dice, IoU, и т.д.)

```python
from src.environment import CropSegmentationEnv

env = CropSegmentationEnv(
    image_patches=train_images,
    mask_patches=train_masks,
    patch_size=256,
    action_mode='patch',  # или 'pixel'
    reward_type='dice',
    num_actions=16  # 4x4 grid
)
```

### 3. RL агент

`PPOAgent` использует алгоритм Proximal Policy Optimization для обучения:

- **Feature Extractor**: CNN или ResNet для извлечения признаков
- **Policy Network**: Предсказывает действия
- **Value Network**: Оценивает состояния
- **PPO**: Стабильное обучение с ограничением изменений политики

```python
from src.agent import PPOAgent

agent = PPOAgent(
    in_channels=4,
    num_actions=16,
    lr=3e-4,
    gamma=0.99,
    clip_epsilon=0.2
)
```

### 4. Система вознаграждений

Доступные функции вознаграждения:

- **Dice Coefficient**: F1-метрика для сегментации
- **IoU (Intersection over Union)**: Jaccard index
- **Pixel Accuracy**: Точность на уровне пикселей
- **Boundary IoU**: Качество границ сегментации
- **Комбинированные метрики**: Взвешенные комбинации

```python
from src.rewards import SegmentationRewards

rewards = SegmentationRewards()
dice = rewards.dice_coefficient(prediction, ground_truth)
iou = rewards.iou_score(prediction, ground_truth)
```

## ⚙️ Конфигурация

### Основные параметры

#### Данные

```yaml
image_path: 'data/raw/satellite_image.tif'
shapefile_path: 'data/raw/crop_boundaries.shp'
patch_size: 256        # Размер патча
stride: 128            # Шаг скользящего окна
```

#### Среда

```yaml
action_mode: 'patch'   # 'patch' или 'pixel'
reward_type: 'dice'    # 'dice', 'iou', 'pixel'
num_actions: 16        # Количество действий (для patch: 4x4 grid)
```

#### Обучение

```yaml
num_epochs: 100
episodes_per_epoch: 10
learning_rate: 3.0e-4
gamma: 0.99            # Дисконт-фактор
clip_epsilon: 0.2      # PPO clipping
```

### Предустановленные конфигурации

1. **default_config.yaml**: Базовая конфигурация
   - Патчи 256x256
   - 100 эпох
   - PPO с стандартными параметрами

2. **fast_training.yaml**: Быстрое обучение
   - Патчи 128x128
   - 20 эпох
   - Меньше действий (3x3 grid)

3. **high_quality.yaml**: Максимальное качество
   - Патчи 512x512
   - 200 эпох
   - ResNet feature extractor
   - Тонкая настройка параметров

## 📊 Мониторинг и визуализация

### Weights & Biases

Включите W&B в конфигурации:

```yaml
use_wandb: true
wandb_project: 'rl_crop_segmentation'
experiment_name: 'my_experiment'
```

### Визуализация результатов

```python
from utils.visualization import plot_training_curves, plot_segmentation_comparison

# График обучения
plot_training_curves(metrics, save_path='training_curves.png')

# Сравнение сегментаций
plot_segmentation_comparison(
    image, ground_truth, prediction,
    metrics={'dice': 0.85, 'iou': 0.75}
)
```

## 🔬 Примеры использования

### Пример 1: Базовое обучение

```python
from src.data_loader import GeoDataLoader
from src.environment import CropSegmentationEnv
from src.agent import PPOAgent

# Загрузка данных
loader = GeoDataLoader(
    image_path="data/raw/satellite.tif",
    shapefile_path="data/raw/boundaries.shp",
    patch_size=256
)
image, mask = loader.load_all()
patches = loader.extract_patches()

# Создание среды
env = CropSegmentationEnv(
    image_patches=[p['image'] for p in patches[:100]],
    mask_patches=[p['mask'] for p in patches[:100]]
)

# Инициализация агента
agent = PPOAgent(in_channels=4, num_actions=16)

# Обучение
for episode in range(1000):
    obs, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, log_prob, value = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.add(obs['image'], action, reward, value, log_prob, done)
        episode_reward += reward

    # Обновление политики
    agent.update()
    print(f"Episode {episode}, Reward: {episode_reward:.4f}")
```

### Пример 2: Оценка модели

```python
from src.agent import PPOAgent
from src.evaluate import Evaluator

# Загрузка обученной модели
agent = PPOAgent(in_channels=4, num_actions=16)
agent.load('experiments/best_model.pt')

# Оценка
evaluator = Evaluator(config, 'experiments/best_model.pt')
metrics = evaluator.evaluate_all()

print(f"Dice: {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
print(f"IoU: {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")

# Визуализация
evaluator.visualize_predictions(n_samples=5)
evaluator.save_full_prediction()
```

### Пример 3: Inference на новом снимке

```python
from src.data_loader import GeoDataLoader
from src.agent import PPOAgent
import rasterio

# Загрузка агента
agent = PPOAgent(in_channels=4, num_actions=16)
agent.load('path/to/best_model.pt')

# Загрузка нового изображения
loader = GeoDataLoader(
    image_path="new_satellite_image.tif",
    shapefile_path=None,  # Нет ground truth
    patch_size=256
)
image = loader.load_image()

# Создание пустого shapefile для совместимости
import geopandas as gpd
from shapely.geometry import Polygon
gdf = gpd.GeoDataFrame(geometry=[Polygon()], crs=loader.crs)
loader.shapefile_path = "dummy.shp"
gdf.to_file("dummy.shp")

mask = loader.create_mask(gdf)
patches = loader.extract_patches()

# Предсказание для каждого патча
predictions = []
for patch in patches:
    env = CropSegmentationEnv(
        image_patches=[patch['image']],
        mask_patches=[patch['mask']]
    )
    obs, _ = env.reset()
    done = False

    while not done:
        action, _, _ = agent.select_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    predictions.append(env.predicted_mask)

# Сохранение результата
# ... (код для объединения патчей и сохранения в GeoTIFF)
```

## 🎯 Режимы работы

### Patch Mode (Рекомендуется)

Изображение делится на сетку подпатчей (например, 4x4). Агент классифицирует каждый подпатч целиком.

**Преимущества:**
- ✅ Быстрое обучение
- ✅ Меньше действий
- ✅ Лучше для крупных полей

**Конфигурация:**
```yaml
action_mode: 'patch'
num_actions: 16  # 4x4 grid
```

### Pixel Mode

Агент классифицирует каждый пиксель отдельно.

**Преимущества:**
- ✅ Максимальная детализация
- ✅ Точные границы

**Недостатки:**
- ❌ Очень долгое обучение
- ❌ Много действий

**Конфигурация:**
```yaml
action_mode: 'pixel'
```

## 📈 Советы по обучению

### 1. Выбор гиперпараметров

- **Размер патча**:
  - Маленький (128): Быстрое обучение, меньше контекста
  - Средний (256): Баланс скорости и качества
  - Большой (512): Больше контекста, дольше обучение

- **Learning rate**:
  - Начните с 3e-4
  - Уменьшите до 1e-4 для стабильности
  - Увеличьте до 5e-4 для быстрого обучения

- **Количество действий** (patch mode):
  - 4 (2x2): Очень быстро, грубая сегментация
  - 16 (4x4): Баланс скорости и детализации
  - 64 (8x8): Детальная сегментация, дольше обучение

### 2. Функции вознаграждения

- **Dice**: Лучше для несбалансированных данных
- **IoU**: Строже к ошибкам
- **Pixel Accuracy**: Может давать ложное чувство хорошего качества

### 3. Мониторинг

Следите за:
- ✅ Рост среднего вознаграждения
- ✅ Стабилизация policy loss
- ✅ Не слишком высокая энтропия (агент становится детерминированным)

## 🐛 Решение проблем

### Проблема: Обучение не сходится

**Решения:**
1. Уменьшите learning rate
2. Увеличьте `clip_epsilon`
3. Попробуйте другую reward function
4. Увеличьте количество episodes_per_epoch

### Проблема: Out of Memory

**Решения:**
1. Уменьшите `patch_size`
2. Уменьшите `batch_size`
3. Используйте меньше патчей для обучения
4. Уменьшите `num_actions`

### Проблема: Низкое качество сегментации

**Решения:**
1. Увеличьте количество эпох
2. Используйте `high_quality.yaml` конфигурацию
3. Попробуйте ResNet feature extractor
4. Увеличьте размер патчей
5. Проверьте качество данных

## 📝 Лицензия

MIT License

## 🤝 Вклад

Приветствуются pull requests! Для крупных изменений сначала откройте issue для обсуждения.

## 📧 Контакты

Для вопросов и предложений: [your-email@example.com]

## 📚 Цитирование

Если вы используете этот код в исследовании, пожалуйста, процитируйте:

```bibtex
@software{rl_crop_segmentation,
  title={RL-based Crop Segmentation Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/rl_segmentation}
}
```

## 🙏 Благодарности

- Stable-Baselines3 за отличную реализацию RL алгоритмов
- Rasterio и GeoPandas за работу с геопространственными данными
- PyTorch за deep learning фреймворк

---

**Удачи с обучением! 🚀**
