"""A collection of tools for single-particle phase retrieval"""
import enum

try:
    import cupy
except ImportError:
    cupy = None


class Backend(enum.Enum):
    """Enum coding for the available backends"""
    CPU = 1
    CUPY = 2


def set_backend(new_backend):
    """Select if calculations are run on CPU or GPU"""
    global backend, numpy, scipy, fft
    global ndimage, real_type, complex_type
    backend = new_backend
    if backend == Backend.CPU:
        import numpy
        import scipy
        from scipy.fft import fft
        from scipy.ndimage import ndimage
        real_type = numpy.float32
        complex_type = numpy.complex64
    elif backend == Backend.CUPY:
        import cupy as numpy
        from cupyx import scipy
        from cupyx.scipy import fft
        from cupyx.scipy import ndimage
        import functools
        real_type = functools.partial(numpy.asarray, dtype="float32")
        complex_type = functools.partial(numpy.asarray, dtype="complex64")
    else:
        raise ValueError(f"Unknown backend {backend}")


def cupy_on():
    """Check fi the backend is Backend.CUPY"""
    return backend == Backend.CUPY


def cpu_on():
    """Check fi the backend is Backend.CPU"""
    return backend == Backend.CPU


def cpu_array(array):
    """Convert an array to standard numpy, regardless of start type"""
    if cupy and isinstance(array, cupy.ndarray):
        return array.get()
    else:
        return array


if cupy and cupy.cuda.is_available():
    set_backend(Backend.CUPY)
    print("Using Cupy backend")
else:
    set_backend(Backend.CPU)
    print("Using CPU backend")


class ModifyAlgorithm:
    """Abstract class for algorithms that work on phasing algorithms"""
    def __init__(self):
        pass


class ThresholdSupport(ModifyAlgorithm):
    """Update the support by thresholding a blurred version of real-space"""
    def __init__(self, threshold, blur):
        super().__init__()
        self.threshold = threshold
        self.blur = blur

    def update_support(self, algorithm):
        """Apply the support update rule for the real space in algorithm"""
        blurred_model = ndimage.gaussian_filter(abs(algorithm.real_model),
                                                self.blur)
        rescaled_threshold = blurred_model.max() * self.threshold
        algorithm.support = blurred_model > rescaled_threshold


class AreaSupport(ModifyAlgorithm):
    """Update the support to the densest part of real-space after blurring"""
    def __init__(self, area, blur):
        super().__init__()
        self.area = area
        self.blur = blur

    def update_support(self, algorithm):
        """Apply the support update rule for the real space in algorithm"""
        blurred_model = ndimage.gaussian_filter(abs(algorithm.real_model),
                                                self.blur)
        area_percent = self.area/numpy.product(blurred_model.shape)
        threshold = numpy.percentile(blurred_model.flat, area_percent)
        algorithm.support = blurred_model > blurred_model.max() * threshold


class CenterSupport(ModifyAlgorithm):
    """Shift the support to the center of the field of view"""
    def __init__(self, array_shape, kernel_sigma):
        super().__init__()
        arrays_1d = [numpy.arange(this_size) - this_size/2.
                     for this_size in array_shape]

        arrays_nd = numpy.meshgrid(*arrays_1d, indexing="ij")
        radius2 = numpy.zeros(array_shape)
        for this_dim in arrays_nd:
            radius2 = radius2 + this_dim**2
        gaussian_kernel = numpy.exp(-radius2/(2.*kernel_sigma**2))
        self._kernel = fft.ifftshift(complex_type(gaussian_kernel))
        self._kernel_ft_conj = numpy.conj(fft.fftn(self._kernel))
        self._shape = array_shape

    def _find_center(self, array):
        conv = fft.ifftn(fft.fftn(fft.fftshift(array))*self._kernel_ft_conj)
        pos = numpy.unravel_index(conv.argmax(), conv.shape)
        return pos
        # return [int(p) for p in pos]

    def update_support(self, algorithm):
        """Apply the support update rule for the real space in algorithm"""
        pos = self._find_center(algorithm.support)
        if backend == Backend.CUPY:
            shift = [-p.get() for p in pos]
        else:
            shift = [-p for p in pos]
        algorithm.support = numpy.roll(algorithm.support, shift=tuple(shift),
                                       axis=tuple(range(len(shift))))


class ConvexOptimizationAlgorithm:
    """Base class for all convex optimization algorithms"""
    def __init__(self, support=None, intensities=None,
                 amplitudes=None, mask=None,
                 real_model=None, fourier_model=None,
                 link=None):

        self._real_model = None
        self._support = None

        if link is None:
            if support is None:
                raise ValueError("Must specify a support")
            self.support = support

            if mask is not None:
                self._mask = fft.fftshift(real_type(mask))
            else:
                self._mask = fft.fftshift(real_type(numpy.ones(support.shape)))

            if intensities is None and amplitudes is None:
                raise ValueError("Algorithm requires either amplitudes "
                                 "or intensities")
            if amplitudes is None:
                amplitudes = numpy.sqrt(real_type(intensities))
            self._amplitudes = fft.fftshift(real_type(amplitudes))

            if real_model is not None and fourier_model is not None:
                raise ValueError("Can not specify both real and fourier model "
                                 "at the same time.")
            if real_model is not None:
                self.real_model = real_model
            elif fourier_model is not None:
                self.fourier_model = fourier_model
            else:
                # If no model is specified initialize with random phases
                shape = self.amplitudes.shape
                phases = 2*numpy.pi*numpy.random.random(shape)
                self.fourier_model = self.amplitudes*numpy.exp(1.j*phases)

            self._iteration = [0]
        else:
            self.link(link)

    def link(self, alg):
        """Make this algorithm share all variables with another algorithm"""
        if not isinstance(alg, ConvexOptimizationAlgorithm):
            raise ValueError("link must be a ConvexOptimizationAlgorithm "
                             "object")
        self._support = alg._support
        self._amplitudes = alg._amplitudes
        self._mask = alg._mask
        self._real_model = alg._real_model
        self._iteration = alg._iteration

    def is_linked(self, alg):
        """Check if two algorithms are properly linked"""
        if (
                id(self._support) == id(alg._support) and
                id(self._amplitudes) == id(alg._amplitudes) and
                id(self._mask) == id(alg._mask) and
                id(self._real_model) == id(alg._real_model)
        ):
            return True
        else:
            return False

    @property
    def iteration(self):
        return self._iteration[0]

    @iteration.setter
    def iteration(self, iteration):
        self._iteration[0] = iteration

    @property
    def real_model(self):
        return fft.ifftshift(self._real_model)

    @real_model.setter
    def real_model(self, real_model):
        if (
                self._real_model is not None and
                self._real_model.shape == real_model.shape
        ):
            self._real_model[...] = complex_type(fft.fftshift(real_model))
        else:
            self._real_model = complex_type(fft.fftshift(real_model))

    def link_model(self, alg):
        """make the real_model ahared between two algorithms"""
        self._real_model = alg._real_model

    @property
    def support(self):
        return fft.ifftshift(self._support)

    @support.setter
    def support(self, support):
        if self._support is not None and self._support.shape == support.shape:
            self._support[...] = real_type(fft.fftshift(support))
        else:
            self._support = real_type(fft.fftshift(support))

    def link_support(self, alg):
        """make the support ahared between two algorithms"""
        self._support = alg._support

    @property
    def fourier_model(self):
        return fft.ifftshift(fft.fftn(self._real_model))

    @fourier_model.setter
    def fourier_model(self, fourier_model):
        if (
                self._real_model is not None and
                self._real_model.shape == fourier_model.shape
        ):
            self._real_model[...] = fft.ifftn(fft.fftshift(fourier_model))
        else:
            self._real_model = fft.ifftn(fft.fftshift(fourier_model))

    @property
    def fourier_model_projected(self):
        fourier_model = fft.fftn(self._real_model)
        phases = numpy.angle(fourier_model)
        projected = (self._amplitudes*numpy.exp(1.j*phases) +
                     (1-self._mask)*fourier_model)
        return fft.ifftshift(projected)
    # return fft.ifftshift(numpy.exp(1.j*phases)*self._amplitudes +
    #                      (1-self._mask)*fft.fftn(self._real_model))

    @property
    def real_model_before_projection(self):
        projected = self.fourier_model_projected()
        return fft.fftshift(fft.ifftn(projected))
        # return fft.fftshift(fft.ifftn(numpy.exp(1.j*phases)*self._amplitudes+
        #                               (1-self._mask)*fourier_model))

    @property
    def amplitudes(self):
        return fft.ifftshift(self._amplitudes)

    @property
    def mask(self):
        return fft.ifftshift(self._mask)

    def fourier_space_constraint(self, data):
        """Apply the Fourier-space constraint to a real-space model"""
        data_ft = fft.fftn(data)
        phases = numpy.angle(data_ft)
        # result = fft.ifftn(self._mask*data_ft/abs(data_ft)*self._amplitudes +
        #                          (1-self._mask)*data_ft)
        result = fft.ifftn(self._mask*self._amplitudes*numpy.exp(1.j*phases) +
                           (1-self._mask)*data_ft)
        return result

    def real_space_constraint(self, data):
        """Apply the real-space constraint to a real-space model"""
        return data*self._support

    def fourier_space_error(self):
        """Fourier-space error of the current iterate"""
        fourier_model = fft.fftn(self._real_model)
        diff = abs(fourier_model*self._mask) - self._amplitudes
        # Rescale to make Fourier and real-space error have
        # similar units
        rescaling = numpy.sqrt(numpy.prod(numpy.array(self._mask.shape)))
        return numpy.sqrt(((diff / rescaling)**2).sum())

    def real_space_error(self):
        """Real-space error of the current iterate"""
        before_projection = self.fourier_space_constraint(self._real_model)
        diff = abs(before_projection*(1-self._support))
        return numpy.sqrt((diff**2).sum())

    @property
    def real_model_projected(self):
        return fft.ifftshift(self._real_model*self._support)

    def iterate(self):
        """Update the iterate. Should be overloaded in subclasses"""
        self._iteration[0] += 1


class ErrorReduction(ConvexOptimizationAlgorithm):
    """Basic error-reduction algorithm. Used mainly for refinement"""
    def iterate(self):
        super().iterate()
        model_fc = self.fourier_space_constraint(self._real_model)
        self._real_model[:] = self._support*model_fc


class RelaxedAveragedAlternatingReflectors(ConvexOptimizationAlgorithm):
    """Relaxed averaged alternating projetors (RAAR) algorithm"""
    def __init__(self, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def iterate(self):
        super().iterate()
        # Rs*Rm+I = (2*Ps-I)*(2*Pm-I)+I = 4*Ps*Pm - 2*Ps - 2*Pm + 2*I
        model_fc = self.fourier_space_constraint(self._real_model)
        self._real_model[:] = (0.5*self.beta *
                               (2.*self._real_model -
                                2.*model_fc -
                                2.*self._real_model*self._support +
                                4.*model_fc*self._support) +
                               (1-self.beta)*model_fc)


class HybridInputOutput(ConvexOptimizationAlgorithm):
    """Hybrid input output (HIO) algorithm"""
    def __init__(self, beta, support=None, intensities=None,
                 amplitudes=None, mask=None, real_model=None,
                 fourier_model=None, link=None):
        super().__init__(support, intensities, amplitudes, mask,
                         real_model, fourier_model, link)
        self.beta = beta

    def iterate(self):
        super().iterate()
        model_fc = self.fourier_space_constraint(self._real_model)
        self._real_model[:] = (self._support*model_fc +
                               (1-self._support)*(self._real_model -
                                                  self.beta*model_fc))


class DifferenceMap(ConvexOptimizationAlgorithm):
    """Difference map algorithm"""
    def __init__(self, beta, gamma_s=None, gamma_m=None,
                 support=None, intensities=None, amplitudes=None, mask=None,
                 real_model=None, fourier_model=None, link=None):
        super().__init__(support, intensities, amplitudes, mask,
                         real_model, fourier_model, link)
        self.beta = beta
        sigma = 0.1
        if gamma_s is None:
            # self.gamma_s = - 1/self.beta
            self.gamma_s = - ((4 + (2+beta)*sigma + beta*sigma**2) /
                              (beta*(4 - sigma + sigma**2)))
        else:
            self.gamma_s = gamma_s
        if gamma_m is None:
            # self.gamma_m = 1/self.beta
            self.gamma_m = ((6 - 2*sigma - beta*(2 - 3*sigma + sigma**2)) /
                            (beta*(4 - sigma + sigma**2)))
        else:
            self.gamma_m = gamma_m

    def iterate(self):
        super().iterate()
        model_fc = self.fourier_space_constraint(self._real_model)
        model_rc = self.real_space_constraint(self._real_model)

        fs = (1+self.gamma_s)*model_rc - self.gamma_s*self._real_model
        fm = (1+self.gamma_m)*model_fc - self.gamma_m*self._real_model
        self._real_model[...] = (self._real_model +
                                 self.beta*(self.real_space_constraint(fm) -
                                            self.fourier_space_constraint(fs)))


def reality_constraint(algorithm):
    """Update an algorithm (class) to add a reality constraint"""
    def iterate(self):
        algorithm.iterate(self)
        self._real_model[:] = numpy.real(self._real_model)
    new_class = type("Real"+algorithm.__name__, (algorithm, ),
                     {"iterate": iterate})
    return new_class


def positivity_constraint(algorithm):
    """Update an algorithm (class) to add a positivity constraint"""
    def iterate(self):
        algorithm.iterate(self)
        real_part = numpy.real(self._real_model)
        real_part[real_part < 0.] = 0.
    new_class = type("Pos"+algorithm.__name__, (algorithm, ),
                     {"iterate": iterate})
    return new_class


class CombineReconstructions:
    """Combine independent reconstructions to calculate average and PRTF"""
    def __init__(self, reference):
        self.reference = complex_type(reference)
        self.fourier_reference = fft.fftn(fft.fftshift(self.reference))
        self.image_sum = numpy.zeros_like(self.reference)
        self.phase_sum = numpy.zeros_like(self.reference)
        self.counter = 0

    def add_image(self, image):
        """Add one reconstruction"""
        image = complex_type(image)
        fourier_image = fft.fftn(fft.fftshift(image))
        conv_1 = fft.ifftn(numpy.conj(fourier_image)*self.fourier_reference)
        conv_2 = fft.ifftn(fourier_image*self.fourier_reference)

        axis_tuple = tuple(range(len(image.shape)))
        if abs(conv_1).max() > abs(conv_2).max():
            trans = numpy.unravel_index(abs(conv_1).argmax(), conv_1.shape)
            translated_image = numpy.roll(image, trans, axis_tuple)
        else:
            trans = numpy.unravel_index(abs(conv_2).argmax(), conv_2.shape)
            trans = tuple(t+1 for t in trans)
            flipped_image = image[(slice(None, None, -1), )*len(trans)]
            translated_image = numpy.roll(flipped_image, trans, axis_tuple)

        ft = fft.ifftshift(fft.fftn(fft.fftshift(translated_image)))
        average_phase = numpy.angle(ft[tuple(s//2 for s in ft.shape)])
        ft /= numpy.exp(1.j*average_phase)
        self.phase_sum += numpy.exp(1.j*numpy.angle(ft))
        self.image_sum += translated_image / numpy.exp(1.j*average_phase)
        self.counter += 1

    def average_image(self):
        """Retrieve the average of the previously added reconstructions"""
        if self.counter <= 0:
            raise ValueError("Trying to retrieve average image without "
                             "adding images first.")
        return self.image_sum / self.counter

    def prtf(self):
        """Retrieve the PRTF of the previously added reconstructions"""
        if self.counter <= 0:
            raise ValueError("Trying to retrieve PRTF without adding "
                             "images first.")
        return abs(self.phase_sum / self.counter)
