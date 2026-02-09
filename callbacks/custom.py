"""
Custom Distributed Data Parallel (DDP) Strategy.

This strategy extends PyTorch Lightning's DDPStrategy and enables
static graph optimization for better performance and reduced overhead
when the computation graph does not change between iterations.

Use this only when your model architecture is fixed across steps.
"""

from pytorch_lightning.strategies import DDPStrategy


class StaticGraphDDPStrategy(DDPStrategy):
    """
    DDP strategy with static graph optimization enabled.

    The key modification is calling `_set_static_graph()` on the wrapped
    model after DDP initialization.
    """

    def configure_ddp(self) -> None:
        """
        Wrap the model with DistributedDataParallel and register hooks.
        Then activate static graph mode for performance gains.
        """

        # Build the DDP-wrapped model
        self._model = self._setup_model(self.model)

        # Register communication hooks (gradient sync, etc.)
        self._register_ddp_hooks()

        # ----------------------------------------------------------
        # 🚀 Performance optimization:
        # Inform DDP that the graph is static across iterations.
        # This avoids repeated graph searches and improves speed.
        # ----------------------------------------------------------
        self._model._set_static_graph()
