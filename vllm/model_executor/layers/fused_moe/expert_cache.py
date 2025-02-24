import time
from typing import Dict, Optional, Tuple
import torch

class ExpertCacheManager:
    """Manages caching of expert weights between CPU and GPU memory."""
    
    def __init__(self, max_cached_experts: int = 32):
        self.max_cached_experts = max_cached_experts
        # Maps expert_idx -> (tensor, last_used_time)
        self._resident_experts: Dict[int, Tuple[torch.Tensor, float]] = {}
        
    def register(self, expert_idx: int, tensor: torch.Tensor) -> None:
        """Register an expert's weight tensor with the cache manager."""
        if tensor.device.type != "cpu":
            # Move to pinned CPU memory if not already there
            cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
            cpu_tensor.copy_(tensor)
            tensor.data = cpu_tensor
            
    def is_resident(self, expert_idx: int) -> bool:
        """Check if an expert's weights are currently in GPU memory."""
        return expert_idx in self._resident_experts
        
    def load_expert(self, expert_idx: int, tensor: torch.Tensor) -> None:
        """Load an expert's weights into GPU memory, evicting if needed."""
        if self.is_resident(expert_idx):
            # Update usage timestamp
            self._resident_experts[expert_idx] = (
                self._resident_experts[expert_idx][0], 
                time.time()
            )
            return

        # If at capacity, evict least recently used
        if len(self._resident_experts) >= self.max_cached_experts:
            lru_idx = min(
                self._resident_experts.items(),
                key=lambda x: x[1][1]
            )[0]
            del self._resident_experts[lru_idx]
            
        # Load expert to GPU
        if tensor.device.type == "cpu":
            gpu_tensor = torch.empty_like(tensor, device="cuda")
            gpu_tensor.copy_(tensor, non_blocking=True)
            self._resident_experts[expert_idx] = (gpu_tensor, time.time())
