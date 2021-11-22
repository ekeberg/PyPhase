import enum


try:
    import cupy
except ImportError:
    cupy = None

class Backend(enum.Enum):
    CPU = 1
    CUPY = 2

def set_backend(new_backend):
    global backend, numpy, scipy, ndimage, real_data_type, complex_data_type
    backend = new_backend
    if backend == Backend.CPU:
        import numpy
        import scipy
        import scipy.ndimage as ndimage
        real_data_type = numpy.float32
        complex_data_type = numpy.complex64
    elif backend == Backend.CUPY:
        import cupy as numpy
        import cupyx.scipy as scipy
        import cupyx.scipy.ndimage as ndimage
        import functools
        real_data_type = functools.partial(numpy.asarray, dtype="float32")
        complex_data_type = functools.partial(numpy.asarray, dtype="complex64")
    else:
        raise ValueError(f"Unknown backend {backend}")

def cupy_on():
    return backend == Backend.CUPY

def cpu_on():
    return backend == Backend.CPU

def cpu_array(array):
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
    def __init__(self):
        pass


class ThresholdSupport(ModifyAlgorithm):
    def __init__(self, threshold, blur):
        self.threshold = threshold
        self.blur = blur

    def update_support(self, algorithm):
        blurred_model = ndimage.gaussian_filter(abs(algorithm.real_model), self.blur)
        algorithm.support = blurred_model > blurred_model.max() * self.threshold


class AreaSupport(ModifyAlgorithm):
    def __init__(self, area, blur):
        self.area = area
        self.blur = blur

    def update_support(self, algorithm):
        blurred_model = ndimage.gaussian_filter(abs(algorithm.real_model), self.blur)
        threshold = numpy.percentile(blurred_model.flat, 100*self.area/numpy.product(blurred_model.shape))
        algorithm.support = blurred_model > blurred_model.max() * threshold
        

class CenterSupport(ModifyAlgorithm):
    def __init__(self, array_shape, kernel_sigma):
        # arrays_1d = [numpy.fft.fftshift(numpy.arange(this_size) - this_size/2. + 0.5)
        #              for this_size in array_shape]
        arrays_1d = [numpy.arange(this_size) - this_size/2.
                     for this_size in array_shape]

        arrays_nd = numpy.meshgrid(*arrays_1d, indexing="ij")
        radius2 = numpy.zeros(array_shape)
        for this_dim in arrays_nd:
            radius2 = radius2 + this_dim**2
        self._gaussian_kernel = numpy.fft.ifftshift(complex_data_type(numpy.exp(-radius2/(2.*kernel_sigma**2))))
        self._gaussian_kernel_ft_conj = numpy.conj(numpy.fft.fftn(self._gaussian_kernel))
        self._shape = array_shape

    def _find_center(self, array):
        conv = numpy.fft.ifftn(numpy.fft.fftn(numpy.fft.fftshift(array))*self._gaussian_kernel_ft_conj)
        pos = numpy.unravel_index(conv.argmax(), conv.shape)
        return pos
        # return [int(p) for p in pos]
    
    def update_support(self, algorithm):
        pos = self._find_center(algorithm.support)
        # for dim_index, shift in enumerate(pos):
        #     algorithm.support = numpy.roll(algorithm.support, -int(shift) % self._shape[dim_index], axis=dim_index)
        if backend == Backend.CUPY:
            shift = [-p.get() for p in pos]
        else:
            shift = [-p for p in pos]
        # print(shift)
        algorithm.support = numpy.roll(algorithm.support, shift=tuple(shift), axis=tuple(range(len(shift))))
        # algorithm._support[...] = numpy.roll(algorithm._support, shift)

        
class ConvexOptimizationAlgorithm:
    def __init__(self, support=None, intensities=None, amplitudes=None, mask=None,
                 real_model=None, fourier_model=None, link=None):

        self._real_model = None
        self._support = None
        
        if link is None:
            if support is None:
                raise ValueError("Must specify a support")
            self.support = support

            if mask is not None:
                self._mask = numpy.fft.fftshift(real_data_type(mask))
            else:
                self._mask = numpy.fft.fftshift(real_data_type(numpy.ones(support.shape)))

            if intensities is None and amplitudes is None:
                raise ValueError("Algorithm requires either amplitudes or intensities")
            if amplitudes is not None:
                self._amplitudes = numpy.fft.fftshift(real_data_type(amplitudes))
            else:
                self._amplitudes = numpy.fft.fftshift(numpy.sqrt(real_data_type(intensities)))

            if real_model is not None and fourier_model is not None:
                raise ValueError("Can not specify both real and fourier model at the same time.")
            if real_model is not None:
                self.real_model = real_model
            elif fourier_model is not None:
                self.fourier_model = fourier_model
            else:
                # If no model is specified initialize with random phases
                self.fourier_model = self.amplitudes*numpy.exp(numpy.pi*2.j*numpy.random.random((self.amplitudes.shape)))
        else:
            self.link(link)
            
    def link(self, alg):
        if not isinstance(alg, ConvexOptimizationAlgorithm):
            raise ValueError("link must be a ConvexOptimizationAlgorithm object")
        self._support = alg._support
        self._amplitudes = alg._amplitudes
        self._mask = alg._mask
        self._real_model = alg._real_model

    def is_linked(self, alg):
        if (self._support == alg._support and self._amplitudes == alg._amplitudes and
            self._mask == alg._mask and self._real_model == alg._real_model):
            return True
        else:
            return False
        
    @property
    def real_model(self):
        return numpy.fft.ifftshift(self._real_model)

    # @property
    # def real_model_cpu(self):
    #     if cupy_on():
    #         return self.real_model.get()
    #     else:
    #         return self.real_model

    @real_model.setter
    def real_model(self, real_model):
        if self._real_model is not  None and self._real_model.shape == real_model.shape:
            self._real_model[...] = complex_data_type(numpy.fft.fftshift(real_model))
        else:
            self._real_model = complex_data_type(numpy.fft.fftshift(real_model))
            
    def link_model(self, alg):
        self._real_model = alg._real_model

    @property
    def support(self):
        return numpy.fft.ifftshift(self._support)

    # @property
    # def support_cpu(self):
    #     if cupy_on():
    #         return self.support.get()
    #     else:
    #         return self.support
    
    @support.setter
    def support(self, support):
        if self._support is not None and self._support.shape == support.shape:
            self._support[...] = real_data_type(numpy.fft.fftshift(support))
        else:
            self._support = real_data_type(numpy.fft.fftshift(support))
    
    def link_support(self, alg):
        self._support = alg._support

    @property
    def fourier_model(self):
        return numpy.fft.ifftshift(numpy.fft.fftn(self._real_model))

    # @property
    # def fourier_model_cpu(self):
    #     if cupy_on():
    #         return self.fourier_model.get()
    #     else:
    #         return self.fourier_model

    @fourier_model.setter
    def fourier_model(self, fourier_model):
        if self._real_model is not None and self._real_model.shape == fourier_model.shape:
            self._real_model[...] = numpy.fft.ifftn(numpy.fft.fftshift(fourier_model))
        else:
            self._real_model = numpy.fft.ifftn(numpy.fft.fftshift(fourier_model))

    @property
    def fourier_model_projected(self):
        return numpy.fft.ifftshift(numpy.exp(1.j*numpy.angle(numpy.fft.fftn(self._real_model)))*self._amplitudes + (1-self._mask)*numpy.fft.fftn(self._real_model))

    @property
    def real_model_before_projection(self):
        return numpy.fft.fftshift(numpy.fft.ifftn(numpy.exp(1.j*numpy.angle(numpy.fft.fftn(self._real_model)))*self._amplitudes + (1-self._mask)*numpy.fft.fftn(self._real_model)))
        
    @property
    def amplitudes(self):
        return numpy.fft.ifftshift(self._amplitudes)
        
    @property
    def mask(self):
        return numpy.fft.ifftshift(self._mask)
        
    def fourier_space_constraint(self, data):
        """Input is real space"""
        data_ft = numpy.fft.fftn(data)
        # result = numpy.fft.ifftn(self._mask*data_ft/abs(data_ft)*self._amplitudes +
        #                          (1-self._mask)*data_ft)
        result = numpy.fft.ifftn(self._mask*numpy.exp(1.j*numpy.angle(data_ft))*self._amplitudes +
                                 (1-self._mask)*data_ft)
        return result

    def real_space_constraint(self, data):
        return data*self._support

    def fourier_space_error(self):
        return numpy.sqrt( (( (abs(numpy.fft.fftn(self._real_model)*self._mask) - self._amplitudes) /
                              numpy.prod(numpy.sqrt(numpy.array(self._amplitudes.shape))) )**2).sum() )

    def real_space_error(self):
        return numpy.sqrt( (abs(self.fourier_space_constraint(self._real_model)*(1-self._support))**2).sum() )

    @property
    def real_model_projected(self):
        return numpy.fft.ifftshift(self._real_model*self._support)

    def iterate(self):
        pass


class ErrorReduction(ConvexOptimizationAlgorithm):
    
    def iterate(self):
        model_fc = self.fourier_space_constraint(self._real_model)
        self._real_model[:] = self._support*model_fc


class RelaxedAveragedAlternatingReflectors(ConvexOptimizationAlgorithm):
    def __init__(self, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def iterate(self):
        # Rs*Rm+I = (2*Ps-I)*(2*Pm-I)+I = 4*Ps*Pm - 2*Ps - 2*Pm + 2*I
        model_fc = self.fourier_space_constraint(self._real_model)
        self._real_model[:] = (0.5*self.beta*(2.*self._real_model -
                                              2.*model_fc -
                                              2.*self._real_model*self._support +
                                              4.*model_fc*self._support) +
                               (1-self.beta)*model_fc)


class HybridInputOutput(ConvexOptimizationAlgorithm):
    def __init__(self, beta, support=None, intensities=None, amplitudes=None, mask=None,
                 real_model=None, fourier_model=None, link=None):
        super().__init__(support, intensities, amplitudes, mask, real_model, fourier_model, link)
        self.beta = beta

    def iterate(self):
        model_fc = self.fourier_space_constraint(self._real_model)
        self._real_model[:] = self._support*model_fc + (1-self._support)*(self._real_model -
                                                                          self.beta*model_fc)


class DifferenceMap(ConvexOptimizationAlgorithm):
    """Does not work yet"""
    def __init__(self, beta, gamma_s, gamma_m,
                 support=None, intensities=None, amplitudes=None, mask=None,
                 real_model=None, fourier_model=None, link=None):
        super().__init__(support, intensities, amplitudes, mask, real_model, fourier_model, link)
        self.beta = beta
        self.gamma_s = gamma_s
        self.gamma_m = gamma_m

    def iterate(self):
        model_fc = self.fourier_space_constraint(self._real_model)
        model_rc = self._real_model*self._support
        # self.real_model[:] = self.support*model_fc + (1-self.support)*(self.real_model -
        #                                                                self.beta*model_fc)
        self.real_model[:] = (self._real_model +
                              self.beta*(1+self.gamma_s)*model_fc*self._support -
                              self.beta*self.gamma_s*self._real_model*self._support -
                              self.beta*self.fourier_space_constraint((1+self.gamma_m)*self._real_model*self._support) -
                              self.beta*self.fourier_space_constraint(-self.gamma_m*self._real_model))


class PosRealHybridInputOutput(ConvexOptimizationAlgorithm):
    def __init__(self, beta, support=None, intensities=None, amplitudes=None, mask=None,
                 real_model=None, fourier_model=None, link=None):
        super().__init__(support, intensities, amplitudes, mask, real_model, fourier_model, link)
        self.beta = beta

    def iterate(self):
        model_fc = self.fourier_space_constraint(self.real_model)
        self._real_model[:] = self._support*model_fc + (1-self._support)*(self._real_model -
                                                                          self.beta*model_fc)
        self._real_model[:] = numpy.real(self._real_model)
        real_part = numpy.real(self._real_model)
        real_part[real_part < 0.] = 0.


class PosRealErrorReduction(ConvexOptimizationAlgorithm):
    
    def iterate(self):
        model_fc = self.fourier_space_constraint(self._real_model)
        self._real_model[:] = self._support*model_fc
        self._real_model[:] = numpy.real(self._real_model)
        real_part = numpy.real(self._real_model)
        real_part[real_part < 0.] = 0.


def add_reality_constraint(algorithm_object):
    import types
    def iterate(self):
        self._before_reality_constraint_iterate()
        self._real_model[:] = numpy.real(self._real_model)
    print(iterate)
    print(algorithm_object.iterate)
    algorithm_object._before_reality_constraint_iterate = algorithm_object.iterate
    algorithm_object.iterate = types.MethodType(iterate, algorithm_object)
    print(algorithm_object.iterate)


def reality_constraint(algorithm):
    def iterate(self):
        algorithm.iterate(self)
        self._real_model[:] = numpy.real(self._real_model)
    new_class = type("Real"+algorithm.__name__, (algorithm, ), {"iterate": iterate})
    return new_class


def positivity_constraint(algorithm):
    def iterate(self):
        algorithm.iterate(self)
        real_part = numpy.real(self._real_model)
        real_part[real_part < 0.] = 0.
    new_class = type("Pos"+algorithm.__name__, (algorithm, ), {"iterate": iterate})
    return new_class


class CombineReconstructions:
    def __init__(self, real_space):
        self.real_space = complex_data_type(real_space)
        self.fourier_space = self._fourier_space(self.real_space)

    @staticmethod
    def _fourier_space(real_space):
        axis_tuple = tuple(range(1, len(real_space.shape)))
        return numpy.fft.ifftshift(numpy.fft.fftn(numpy.fft.fftshift(real_space, axes=axis_tuple),
                                                  axes=axis_tuple), axes=axis_tuple)

    def center_images(self, template):
        # Update fourier space
        self.fourier_space[:] = self._fourier_space(self.real_space)
        ax = tuple(range(1, len(self.fourier_space.shape)))
        template_fft_conj = numpy.conj(numpy.fft.ifftshift(numpy.fft.fftn(numpy.fft.fftshift(template))))
        conv_1  = numpy.fft.ifftshift(numpy.fft.ifftn(numpy.fft.fftshift(
            self.fourier_space*template_fft_conj[numpy.newaxis, :, :], axes=ax), axes=ax), axes=ax)
        conv_2  = numpy.fft.ifftshift(numpy.fft.ifftn(numpy.fft.fftshift(
            numpy.conj(self.fourier_space)*template_fft_conj[numpy.newaxis, :, :], axes=ax), axes=ax), axes=ax)

        for this_real_space, this_conv_1, this_conv_2 in zip(self.real_space, conv_1, conv_2):
            if abs(this_conv_1).max() >= abs(this_conv_2).max():
                # Don't flip image
                trans = numpy.unravel_index(abs(this_conv_1).argmax(), this_conv_1.shape)
                trans = tuple(-t for t in trans)
                this_real_space[:] = numpy.roll(this_real_space, trans,
                                                axis=tuple(range(len(trans))))
            else:
                # Flip image
                trans = numpy.unravel_index(abs(this_conv_2).argmax(), this_conv_2.shape)
                # trans = tuple(-t for t in trans)
                trans = tuple(t-1 for t in trans)
                this_real_space[:] = numpy.roll(this_real_space, trans,
                                                axis=tuple(range(len(trans))))[::-1, ::-1]

        # Align phases so that the center phase is 0
        self.real_space[:] /= self.fourier_space[(slice(None), ) + tuple(s//2 for s in self.fourier_space.shape[1:])][:, numpy.newaxis, numpy.newaxis]

    def average_image(self):
        return self.real_space.mean(axis=0)

    def prtf(self):
        self.fourier_space[:] = self._fourier_space(self.real_space)
        prtf = abs((self.fourier_space / abs(self.fourier_space)).mean(axis=0))
        return prtf
        
class CombineReconstructionsSequential:
    def __init__(self, reference):
        self.reference = complex_data_type(reference)
        # self.fourier_reference = numpy.fft.ifftshift(numpy.fft.fftn(numpy.fft.fftshift(self.reference)))
        self.fourier_reference = numpy.fft.fftn(numpy.fft.fftshift(self.reference))
        self.image_sum = numpy.zeros_like(self.reference)
        self.phase_sum = numpy.zeros_like(self.reference)
        self.counter = 0

    def add_image(self, image):
        # fourier_image = numpy.fft.ifftshift(numpy.fft.fftn(numpy.fft.fftshift(image)))
        fourier_image = numpy.fft.fftn(numpy.fft.fftshift(image))
        conv_1 = numpy.fft.ifftn(numpy.conj(fourier_image)*self.fourier_reference)
        conv_2 = numpy.fft.ifftn(fourier_image*self.fourier_reference)

        if abs(conv_1).max() > abs(conv_2).max():
            trans = numpy.unravel_index(abs(conv_1).argmax(), conv_1.shape)
            translated_image = numpy.roll(image, trans, tuple(range(len(trans))))
        else:
            trans = numpy.unravel_index(abs(conv_2).argmax(), conv_2.shape)
            # trans = tuple(t-1 for t in trans)
            trans = tuple(t+1 for t in trans)
            #translated_image = numpy.roll(image, trans, tuple(range(len(trans))))[::-1, ::-1]
            flipped_image = image[(slice(None, None, -1), )*len(trans)]
            translated_image = numpy.roll(flipped_image, trans, tuple(range(len(trans))))

        self.image_sum += translated_image
        ft = numpy.fft.ifftshift(numpy.fft.fftn(numpy.fft.fftshift(translated_image)))
        # ft /= ft[ft.shape[0]//2, ft.shape[0]//2]
        ft /= ft[tuple(s//2 for s in ft.shape)]
        self.phase_sum += numpy.exp(1.j*numpy.angle(ft))
        self.counter += 1

    def average_image(self):
        if self.counter <= 0:
            raise ValueError("Trying to retrieve average image without adding images first.")
        return self.image_sum / self.counter

    def prtf(self):
        if self.counter <= 0:
            raise ValueError("Trying to retrieve PRTF without adding images first.")
        return abs(self.phase_sum / self.counter)
        
