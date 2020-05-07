# import afnumpy as numpy
# import afnumpy.fft
# numpy.fft = afnumpy.fft
import numpy

real_data_type = numpy.float32
complex_data_type = numpy.complex64

class ModifyAlgorithm:
    def __init__(self):
        pass


class ThresholdSupport(ModifyAlgorithm):
    def __init__(self, threshold, blur):
        self.threshold = threshold
        self.blur = blur

    def update_support(self, algorithm):
        import scipy.ndimage
        blurred_model = scipy.ndimage.gaussian_filter(abs(algorithm.real_model), self.blur)
        algorithm.support[:] = blurred_model > blurred_model.max() * self.threshold


class CenterSupport(ModifyAlgorithm):
    def __init__(self, array_shape, kernel_sigma):
        arrays_1d = [numpy.fft.fftshift(numpy.arange(this_size) - this_size/2. + 0.5)
                     for this_size in array_shape]
        arrays_nd = numpy.meshgrid(*arrays_1d, indexing="ij")
        radius2 = numpy.zeros(array_shape)
        for this_dim in arrays_nd:
            radius2 = radius2 + this_dim**2
        self._gaussian_kernel = complex_data_type(numpy.exp(-radius2/(2.*kernel_sigma**2)))
        self._gaussian_kernel_ft_conj = numpy.conj(numpy.fft.fftn(self._gaussian_kernel))
        self._shape = array_shape

    def _find_center(self, array):
        conv = numpy.fft.ifftn(numpy.fft.fftn(array)*self._gaussian_kernel_ft_conj)
        pos = numpy.unravel_index(conv.argmax(), conv.shape)
        return pos
    
    def update_support(self, algorithm):
        pos = self._find_center(algorithm.support)
        for dim_index, shift in enumerate(pos):
            algorithm.support[:] = numpy.roll(algorithm.support, -shift % self._shape[dim_index], axis=dim_index)

        
class ConvexOptimizationAlgorithm:
    def __init__(self, support=None, intensities=None, amplitudes=None, mask=None,
                 real_model=None, fourier_model=None):
        if support is None:
            raise ValueError("Must specify a support")
        self.support = real_data_type(support)

        if mask is not None:
            self.mask = real_data_type(mask)
        else:
            self.mask = real_data_type(numpy.ones(support.shape))

        if intensities is None and amplitudes is None:
            raise ValueError("Algorithm requires either amplitudes or intensities")
        if amplitudes is not None:
            self.amplitudes = real_data_type(amplitudes)
        else:
            self.amplitudes = numpy.sqrt(real_data_type(intensities))

        if real_model is not None and fourier_model is not None:
            raise ValueError("Can not specify both real and fourier model at the same time.")
        if real_model is not None:
            self.set_real_model(real_model)
        elif fourier_model is not None:
            self.set_fourier_model(fourier_model)
        else:
            # If no model is specified initialize with random phases
            self.set_fourier_model(self.amplitudes*numpy.exp(numpy.pi*2.j*numpy.random.random((self.amplitudes.shape))))

    # @property
    # def real_model(self):
    #     return numpy.fft.ifftshift(self._real_model)

    def set_real_model(self, real_model, link=False):
        if not link:
            self.real_model = complex_data_type(real_model.copy())
        else:
            if real_model.dtype == complex_data_type:
                self.real_model = real_model
            else:
                raise ValueError("Can not set real_model as link. Type {} doesn't match {}"
                                 .format(str(numpy.dtype(real_model.dtype)),
                                         str(numpy.dtype(complex_data_type))))

    def set_support(self, support, link=False):
        if not link:
            self.support = real_data_type(support.copy())
        else:
            if support.dtype == real_data_type:
                self.support = support
            else:
                raise ValueError("Can not set support as link. Type {} doesn't match {}"
                                 .format(str(numpy.dtype(support.dtype)),
                                         str(numpy.dtype(real_data_type))))
            
    def set_fourier_model(self, fourier_model):
        self.real_model = numpy.fft.fftn(complex_data_type(fourier_model))

    def fourier_space_constraint(self, data):
        """Input is real space"""
        data_ft = numpy.fft.fftn(data)
        result = numpy.fft.ifftn(self.mask*data_ft/abs(data_ft)*self.amplitudes +
                                 (1-self.mask)*data_ft)
        return result

    def real_space_constraint(self, data):
        return data*self.support

    def fourier_space_error(self):
        return numpy.sqrt( (( (abs(numpy.fft.fftn(self.real_model)*self.mask) - self.amplitudes) /
                              numpy.prod(numpy.sqrt(numpy.array(self.amplitudes.shape))) )**2).sum() )

    def real_space_error(self):
        return numpy.sqrt( (abs(self.fourier_space_constraint(self.real_model)*(1-self.support))**2).sum() )

    @property
    def real_model_projected(self):
        return self.real_model*self.support

    def iterate(self):
        pass


class ErrorReduction(ConvexOptimizationAlgorithm):
    
    def iterate(self):
        model_fc = self.fourier_space_constraint(self.real_model)
        self.real_model[:] = self.support*model_fc


class RelaxedAveragedAlternatingReflectors(ConvexOptimizationAlgorithm):
    def __init__(self, beta, support=None, intensities=None, amplitudes=None, mask=None,
                 real_model=None, fourier_model=None):
        super().__init__(support, intensities, amplitudes, mask, real_model, fourier_model)
        self.beta = beta

    def iterate(self):
        # Rs*Rm+I = (2*Ps-I)*(2*Pm-I)+I = 4*Ps*Pm - 2*Ps - 2*Pm + 2*I
        model_fc = self.fourier_space_constraint(self.real_model)
        self.real_model[:] = (0.5*self.beta*(2.*self.real_model -
                                             2.*model_fc -
                                             2.* self.real_model*self.support +
                                             4.*model_fc*self.support) +
                              (1-self.beta)*model_fc)


class HybridInputOutput(ConvexOptimizationAlgorithm):
    def __init__(self, beta, support=None, intensities=None, amplitudes=None, mask=None,
                 real_model=None, fourier_model=None):
        super().__init__(support, intensities, amplitudes, mask, real_model, fourier_model)
        self.beta = beta

    def iterate(self):
        model_fc = self.fourier_space_constraint(self.real_model)
        self.real_model[:] = self.support*model_fc + (1-self.support)*(self.real_model -
                                                                       self.beta*model_fc)


class DifferenceMap(ConvexOptimizationAlgorithm):
    """Does not work yet"""
    def __init__(self, beta, gamma_s, gamma_m,
                 support=None, intensities=None, amplitudes=None, mask=None,
                 real_model=None, fourier_model=None):
        super().__init__(support, intensities, amplitudes, mask, real_model, fourier_model)
        self.beta = beta
        self.gamma_s = gamma_s
        self.gamma_m = gamma_m

    def iterate(self):
        model_fc = self.fourier_space_constraint(self.real_model)
        model_rc = self.real_model*self.support
        # self.real_model[:] = self.support*model_fc + (1-self.support)*(self.real_model -
        #                                                                self.beta*model_fc)
        self.real_model[:] = (self.real_model +
                              self.beta*(1+self.gamma_s)*model_fc*self.support -
                              self.beta*self.gamma_s*self.real_model*self.support -
                              self.beta*self.fourier_space_constraint((1+self.gamma_m)*self.real_model*self.support) -
                              self.beta*self.fourier_space_constraint(-self.gamma_m*self.real_model))


class PosRealHybridInputOutput(ConvexOptimizationAlgorithm):
    def __init__(self, beta, support=None, intensities=None, amplitudes=None, mask=None,
                 real_model=None, fourier_model=None):
        super().__init__(support, intensities, amplitudes, mask, real_model, fourier_model)
        self.beta = beta

    def iterate(self):
        model_fc = self.fourier_space_constraint(self.real_model)
        self.real_model[:] = self.support*model_fc + (1-self.support)*(self.real_model -
                                                                       self.beta*model_fc)
        self.real_model[:] = numpy.real(self.real_model)
        real_part = numpy.real(self.real_model)
        real_part[real_part < 0.] = 0.


class PosRealErrorReduction(ConvexOptimizationAlgorithm):
    
    def iterate(self):
        model_fc = self.fourier_space_constraint(self.real_model)
        self.real_model[:] = self.support*model_fc
        self.real_model[:] = numpy.real(self.real_model)
        real_part = numpy.real(self.real_model)
        real_part[real_part < 0.] = 0.


def add_reality_constraint(algorithm_object):
    import types
    def iterate(self):
        self._before_reality_constraint_iterate()
        self.real_model[:] = numpy.real(self.real_model)
    print(iterate)
    print(algorithm_object.iterate)
    algorithm_object._before_reality_constraint_iterate = algorithm_object.iterate
    algorithm_object.iterate = types.MethodType(iterate, algorithm_object)
    print(algorithm_object.iterate)


def reality_constraint(algorithm):
    def iterate(self):
        algorithm.iterate(self)
        self.real_model[:] = numpy.real(self.real_model)
    new_class = type("Real"+algorithm.__name__, (algorithm, ), {"iterate": iterate})
    return new_class


def positivity_constraint(algorithm):
    def iterate(self):
        algorithm.iterate(self)
        real_part = numpy.real(self.real_model)
        real_part[real_part < 0.] = 0.
    new_class = type("Pos"+algorithm.__name__, (algorithm, ), {"iterate": iterate})
    return new_class


class CombineReconstructions:
    def __init__(self, real_space):
        self.real_space = complex_data_type(real_space)
        self.fourier_space = self._fourier_space(self.real_space)

    @staticmethod
    def _fourier_space(real_space):
        return numpy.fft.fftn(real_space, axes=tuple(range(1, len(real_space.shape))))

    def center_images(self, template):
        # Update fourier space
        self.fourier_space[:] = self._fourier_space(self.real_space)
        ax = tuple(range(1, len(self.fourier_space.shape)))
        conv_1  = numpy.fft.ifftn(self.fourier_space*
                                  numpy.conj(numpy.fft.fftn(template[numpy.newaxis, : , :])),
                                  axes=ax)
        conv_2  = numpy.fft.ifftn(numpy.conj(self.fourier_space)*
                                  numpy.conj(numpy.fft.fftn(template[numpy.newaxis, : , :])),
                                  axes=ax)

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
        self.real_space[:] /= self.fourier_space[:, 0, 0][:, numpy.newaxis, numpy.newaxis]

    def avg_image(self):
        return self.real_space.mean(axis=0)

    def prtf(self):
        self.fourier_space[:] = self._fourier_space(self.real_space)
        prtf = abs((self.fourier_space / abs(self.fourier_space)).mean(axis=0))
        return prtf
        
