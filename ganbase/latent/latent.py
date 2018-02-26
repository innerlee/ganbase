class Latent:
    """
    latent space interface
    """

    def __init__(self):
        #region yapf: disable
        self.intrinsic_dim = 0
        self.embed_in_dim  = 0
        self.embed_out_dim = 0
        self.draw_dim      = 0
        self.R             = 0
        self.confine       = 'hard'
        self.outactivation = ''
        self.eps           = 1e-8
        self.reg_loss      = None
        #endregion yapf: enable

    def sample(self, n, confine='hard'):
        """
        random sampling
        the output has already embeded
        """
        raise NotImplementedError()

    def embed(self, z):
        """
        embed raw input of `z` to ambient space
        """
        raise NotImplementedError()

    def neighbor(self, z, sigma, multiple=1):
        """
        random neighbor, given radius sigma
        the output has already embeded
        """
        raise NotImplementedError()

    def embed_draw(self, z, sigma):
        """
        embed raw input of `z` to drawing space
        """
        raise NotImplementedError()

    def embed_to_draw(self, z, sigma):
        """
        embeded points to drawable points
        """
        raise NotImplementedError()
