"""
Active Learning для умной аннотации данных.

Агент сам определяет, какие регионы требуют человеческой аннотации,
минимизируя затраты на разметку.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy


class UncertaintyEstimation:
    """
    Методы оценки неопределенности для active learning.
    """

    @staticmethod
    def mc_dropout_uncertainty(model: nn.Module,
                               image: torch.Tensor,
                               n_samples: int = 20) -> torch.Tensor:
        """
        Monte Carlo Dropout для оценки неопределенности.

        Args:
            model: Model with dropout
            image: Input image (B, C, H, W)
            n_samples: Number of MC samples

        Returns:
            Uncertainty map (B, 1, H, W)
        """
        model.train()  # Enable dropout

        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = torch.sigmoid(model(image))
                predictions.append(pred)

        predictions = torch.stack(predictions)  # (n_samples, B, 1, H, W)

        # Compute variance as uncertainty
        uncertainty = predictions.var(dim=0)

        model.eval()
        return uncertainty

    @staticmethod
    def predictive_entropy(predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive entropy as uncertainty.

        Args:
            predictions: Probability predictions (B, 1, H, W)

        Returns:
            Entropy map (B, 1, H, W)
        """
        # Binary entropy
        p = predictions.clamp(1e-6, 1 - 1e-6)
        ent = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        return ent

    @staticmethod
    def bald_uncertainty(predictions_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Bayesian Active Learning by Disagreement (BALD).

        Args:
            predictions_list: List of predictions from multiple models/samples

        Returns:
            BALD uncertainty (B, 1, H, W)
        """
        predictions = torch.stack(predictions_list)  # (N, B, 1, H, W)

        # Mean prediction
        mean_pred = predictions.mean(dim=0)

        # Entropy of mean
        entropy_mean = UncertaintyEstimation.predictive_entropy(mean_pred)

        # Mean of entropies
        entropies = [UncertaintyEstimation.predictive_entropy(p) for p in predictions]
        mean_entropy = torch.stack(entropies).mean(dim=0)

        # BALD = Mutual Information = Entropy(mean) - Mean(entropy)
        bald = entropy_mean - mean_entropy

        return bald


class ActiveLearningSelector:
    """
    Выбирает регионы для аннотации на основе различных стратегий.
    """

    def __init__(self, budget: int = 100):
        """
        Initialize active learning selector.

        Args:
            budget: Annotation budget (number of patches/pixels to annotate)
        """
        self.budget = budget
        self.annotated_indices = set()

    def uncertainty_sampling(self,
                            uncertainty_map: torch.Tensor,
                            k: int = None) -> List[Tuple[int, int]]:
        """
        Select regions with highest uncertainty.

        Args:
            uncertainty_map: Uncertainty map (H, W)
            k: Number of regions to select (None = use budget)

        Returns:
            List of (y, x) coordinates
        """
        if k is None:
            k = self.budget

        uncertainty_map = uncertainty_map.cpu().numpy()
        H, W = uncertainty_map.shape

        # Flatten and get top-k
        flat_uncertainty = uncertainty_map.flatten()
        top_k_indices = np.argpartition(flat_uncertainty, -k)[-k:]

        # Convert to coordinates
        coordinates = []
        for idx in top_k_indices:
            y = idx // W
            x = idx % W
            coordinates.append((int(y), int(x)))

        return coordinates

    def diversity_sampling(self,
                          embeddings: torch.Tensor,
                          k: int = None) -> List[int]:
        """
        Select diverse samples using k-means++ like strategy.

        Args:
            embeddings: Sample embeddings (N, D)
            k: Number of samples to select

        Returns:
            List of selected indices
        """
        if k is None:
            k = min(self.budget, len(embeddings))

        embeddings = embeddings.cpu().numpy()
        N = len(embeddings)

        selected = []
        remaining = set(range(N)) - self.annotated_indices

        # Random first sample
        first = np.random.choice(list(remaining))
        selected.append(first)
        remaining.remove(first)

        # Greedily select most diverse
        for _ in range(k - 1):
            if len(remaining) == 0:
                break

            # Compute distances to selected samples
            selected_embeddings = embeddings[selected]
            remaining_embeddings = embeddings[list(remaining)]

            # Distance to nearest selected
            distances = np.min(
                np.linalg.norm(
                    remaining_embeddings[:, None] - selected_embeddings[None, :],
                    axis=2
                ),
                axis=1
            )

            # Select most distant
            most_distant_idx = np.argmax(distances)
            actual_idx = list(remaining)[most_distant_idx]

            selected.append(actual_idx)
            remaining.remove(actual_idx)

        return selected

    def representative_sampling(self,
                               embeddings: torch.Tensor,
                               k: int = None) -> List[int]:
        """
        Select representative samples using clustering.

        Args:
            embeddings: Sample embeddings (N, D)
            k: Number of clusters/samples

        Returns:
            List of selected indices (cluster centers)
        """
        if k is None:
            k = min(self.budget, len(embeddings))

        from sklearn.cluster import KMeans

        embeddings_np = embeddings.cpu().numpy()

        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings_np)

        # Select point closest to each cluster center
        selected = []
        for i in range(k):
            cluster_points = np.where(labels == i)[0]

            if len(cluster_points) == 0:
                continue

            # Find closest to center
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(embeddings_np[cluster_points] - center, axis=1)
            closest_idx = cluster_points[np.argmin(distances)]

            selected.append(int(closest_idx))

        return selected

    def combined_strategy(self,
                         uncertainty_map: torch.Tensor,
                         embeddings: torch.Tensor,
                         k: int = None,
                         uncertainty_weight: float = 0.5) -> List[int]:
        """
        Combine uncertainty and diversity sampling.

        Args:
            uncertainty_map: Uncertainty scores (N,)
            embeddings: Sample embeddings (N, D)
            k: Number of samples
            uncertainty_weight: Weight for uncertainty vs diversity

        Returns:
            List of selected indices
        """
        if k is None:
            k = min(self.budget, len(embeddings))

        # Get top uncertain samples
        n_uncertain = max(1, int(k * 1.5))  # Sample more for diversity filtering
        uncertainty_scores = uncertainty_map.cpu().numpy()
        top_uncertain = np.argsort(uncertainty_scores)[-n_uncertain:]

        # Among uncertain samples, select diverse ones
        uncertain_embeddings = embeddings[top_uncertain]

        # Run diversity sampling on uncertain samples
        # Simple version: compute pairwise distances and select diverse subset
        n_select = k
        selected_local = []
        remaining = list(range(len(top_uncertain)))

        # Random first
        first = np.random.choice(remaining)
        selected_local.append(first)
        remaining.remove(first)

        # Greedy diverse selection
        uncertain_emb_np = uncertain_embeddings.cpu().numpy()

        for _ in range(n_select - 1):
            if len(remaining) == 0:
                break

            selected_embs = uncertain_emb_np[selected_local]
            remaining_embs = uncertain_emb_np[remaining]

            # Distance to nearest selected
            distances = np.min(
                np.linalg.norm(
                    remaining_embs[:, None] - selected_embs[None, :],
                    axis=2
                ),
                axis=1
            )

            # Combine with uncertainty
            remaining_uncertainties = uncertainty_scores[top_uncertain[remaining]]
            scores = (uncertainty_weight * remaining_uncertainties +
                     (1 - uncertainty_weight) * distances / distances.max())

            # Select best score
            best_local_idx = np.argmax(scores)
            best_idx = remaining[best_local_idx]

            selected_local.append(best_idx)
            remaining.remove(best_idx)

        # Convert back to global indices
        selected_global = [int(top_uncertain[i]) for i in selected_local]

        return selected_global


class QueryStrategy(nn.Module):
    """
    Learnable query strategy - агент учится, что запрашивать.

    Вместо эвристик, нейросеть учится предсказывать,
    какие примеры принесут наибольшую пользу при аннотации.
    """

    def __init__(self, feature_dim: int = 512):
        """
        Initialize query strategy network.

        Args:
            feature_dim: Feature dimension
        """
        super().__init__()

        # Network that predicts "query value" for each sample
        self.query_network = nn.Sequential(
            nn.Linear(feature_dim + 3, 256),  # +3 for uncertainty, diversity, representativeness
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self,
                features: torch.Tensor,
                uncertainty: torch.Tensor,
                diversity_score: torch.Tensor,
                representativeness: torch.Tensor) -> torch.Tensor:
        """
        Predict query value for each sample.

        Args:
            features: Sample features (N, feature_dim)
            uncertainty: Uncertainty scores (N, 1)
            diversity_score: Diversity scores (N, 1)
            representativeness: Representativeness scores (N, 1)

        Returns:
            Query values (N, 1) - higher = more valuable to annotate
        """
        # Concatenate all information
        x = torch.cat([features, uncertainty, diversity_score, representativeness], dim=1)

        # Predict query value
        query_value = self.query_network(x)

        return query_value

    def select_samples(self,
                      features: torch.Tensor,
                      uncertainty: torch.Tensor,
                      diversity_score: torch.Tensor,
                      representativeness: torch.Tensor,
                      k: int) -> List[int]:
        """
        Select k samples with highest query value.

        Args:
            features: Sample features
            uncertainty: Uncertainty scores
            diversity_score: Diversity scores
            representativeness: Representativeness scores
            k: Number of samples to select

        Returns:
            List of selected indices
        """
        query_values = self.forward(features, uncertainty, diversity_score, representativeness)

        # Select top-k
        top_k_values, top_k_indices = torch.topk(query_values.squeeze(), k)

        return top_k_indices.cpu().tolist()


class HumanInTheLoop:
    """
    Human-in-the-loop интерфейс.

    Агент запрашивает аннотации для сложных случаев,
    получает feedback и улучшается.
    """

    def __init__(self, annotation_cost: float = 1.0):
        """
        Initialize human-in-the-loop.

        Args:
            annotation_cost: Cost per annotation (for budget tracking)
        """
        self.annotation_cost = annotation_cost
        self.total_cost = 0
        self.annotation_history = []

    def request_annotation(self,
                          image: np.ndarray,
                          uncertainty_map: np.ndarray,
                          current_prediction: np.ndarray) -> Dict:
        """
        Request annotation from human.

        Args:
            image: Image to annotate
            uncertainty_map: Uncertainty map highlighting difficult regions
            current_prediction: Current model prediction

        Returns:
            Annotation dict with ground truth mask
        """
        # In real system, this would show UI to human annotator
        # For simulation, we'll return synthetic annotation

        print(f"Requesting annotation...")
        print(f"  Image shape: {image.shape}")
        print(f"  Max uncertainty: {uncertainty_map.max():.3f}")
        print(f"  Current pred quality: {current_prediction.mean():.3f}")

        # Simulate human annotation (in real system, human provides this)
        # For now, just return the current prediction with some noise
        gt_mask = (current_prediction > 0.5).astype(np.float32)

        annotation = {
            'mask': gt_mask,
            'quality': 1.0,  # Annotation quality score
            'time': 60.0,    # Time spent (seconds)
            'cost': self.annotation_cost
        }

        self.total_cost += annotation['cost']
        self.annotation_history.append(annotation)

        return annotation

    def adaptive_annotation(self,
                           image: np.ndarray,
                           uncertainty_map: np.ndarray,
                           threshold: float = 0.7) -> Dict:
        """
        Request annotation only for highly uncertain regions.

        Args:
            image: Image
            uncertainty_map: Uncertainty map
            threshold: Uncertainty threshold

        Returns:
            Partial annotation (only uncertain regions)
        """
        # Identify highly uncertain regions
        uncertain_mask = uncertainty_map > threshold

        # Request annotation only for these regions
        if uncertain_mask.sum() > 0:
            print(f"Requesting partial annotation for {uncertain_mask.sum()} uncertain pixels")

            # In real system, show only uncertain regions to annotator
            # Simulated annotation
            partial_annotation = {
                'mask': uncertain_mask.astype(np.float32),
                'region_mask': uncertain_mask,
                'cost': self.annotation_cost * (uncertain_mask.sum() / uncertain_mask.size)
            }

            self.total_cost += partial_annotation['cost']
            return partial_annotation

        return None

    def get_statistics(self) -> Dict:
        """Get annotation statistics."""
        return {
            'total_cost': self.total_cost,
            'num_annotations': len(self.annotation_history),
            'avg_time': np.mean([a['time'] for a in self.annotation_history]) if self.annotation_history else 0
        }


class CurriculumActiveLearning:
    """
    Curriculum Active Learning - начинаем с простых примеров,
    постепенно усложняем.
    """

    def __init__(self, initial_difficulty: float = 0.3):
        """
        Initialize curriculum active learning.

        Args:
            initial_difficulty: Starting difficulty level
        """
        self.current_difficulty = initial_difficulty
        self.difficulty_increment = 0.1
        self.performance_history = []

    def get_difficulty_range(self) -> Tuple[float, float]:
        """
        Get current difficulty range.

        Returns:
            Tuple of (min_difficulty, max_difficulty)
        """
        return (
            max(0.0, self.current_difficulty - 0.1),
            min(1.0, self.current_difficulty + 0.1)
        )

    def filter_by_difficulty(self,
                            samples: List[Dict],
                            difficulty_scores: np.ndarray) -> List[int]:
        """
        Filter samples by current difficulty level.

        Args:
            samples: List of samples
            difficulty_scores: Difficulty score for each sample (0-1)

        Returns:
            Indices of samples within difficulty range
        """
        min_diff, max_diff = self.get_difficulty_range()

        valid_indices = []
        for i, score in enumerate(difficulty_scores):
            if min_diff <= score <= max_diff:
                valid_indices.append(i)

        return valid_indices

    def update_curriculum(self, performance: float):
        """
        Update curriculum based on performance.

        Args:
            performance: Current performance metric (0-1)
        """
        self.performance_history.append(performance)

        # If performing well, increase difficulty
        if len(self.performance_history) >= 3:
            recent_performance = np.mean(self.performance_history[-3:])

            if recent_performance > 0.8:
                self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_increment)
                print(f"Increasing difficulty to {self.current_difficulty:.2f}")
            elif recent_performance < 0.5:
                self.current_difficulty = max(0.0, self.current_difficulty - self.difficulty_increment)
                print(f"Decreasing difficulty to {self.current_difficulty:.2f}")


if __name__ == "__main__":
    # Test active learning
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Testing Active Learning Components...")
    print("=" * 60)

    # Test uncertainty estimation
    print("\n1. Testing Uncertainty Estimation...")
    predictions = torch.rand(4, 1, 256, 256).to(device)
    entropy_unc = UncertaintyEstimation.predictive_entropy(predictions)
    print(f"   Entropy uncertainty shape: {entropy_unc.shape}")
    print(f"   Mean uncertainty: {entropy_unc.mean():.4f}")

    # Test active learning selector
    print("\n2. Testing Active Learning Selector...")
    selector = ActiveLearningSelector(budget=50)

    uncertainty_map = torch.rand(256, 256)
    selected_coords = selector.uncertainty_sampling(uncertainty_map, k=10)
    print(f"   Selected {len(selected_coords)} regions")
    print(f"   Sample coordinates: {selected_coords[:3]}")

    # Test learnable query strategy
    print("\n3. Testing Learnable Query Strategy...")
    query_strategy = QueryStrategy(feature_dim=512).to(device)

    features = torch.randn(100, 512).to(device)
    uncertainty = torch.rand(100, 1).to(device)
    diversity = torch.rand(100, 1).to(device)
    representativeness = torch.rand(100, 1).to(device)

    query_values = query_strategy(features, uncertainty, diversity, representativeness)
    selected = query_strategy.select_samples(features, uncertainty, diversity, representativeness, k=10)

    print(f"   Query values shape: {query_values.shape}")
    print(f"   Selected {len(selected)} samples")
    print(f"   Sample indices: {selected[:5]}")

    # Test human-in-the-loop
    print("\n4. Testing Human-in-the-Loop...")
    hitl = HumanInTheLoop(annotation_cost=1.0)

    image = np.random.rand(3, 256, 256)
    unc_map = np.random.rand(256, 256)
    pred = np.random.rand(256, 256)

    annotation = hitl.request_annotation(image, unc_map, pred)
    print(f"   Annotation received: {annotation['mask'].shape}")
    print(f"   Total cost: ${hitl.total_cost:.2f}")

    # Test curriculum active learning
    print("\n5. Testing Curriculum Active Learning...")
    curriculum = CurriculumActiveLearning(initial_difficulty=0.3)

    difficulty_range = curriculum.get_difficulty_range()
    print(f"   Current difficulty range: {difficulty_range}")

    # Simulate performance updates
    for i, perf in enumerate([0.6, 0.7, 0.85, 0.9]):
        curriculum.update_curriculum(perf)
        print(f"   After {i+1} updates: difficulty = {curriculum.current_difficulty:.2f}")

    print("\n" + "=" * 60)
    print("✓ All active learning components working!")
