import kornia as K

# define differentiable augmentation library and intialize them
class _Jitter(K.augmentation.ColorJitter):
    def __init__(self, brightness, contrast, saturation, hue):
        super(_Jitter, self).__init__( brightness, contrast, saturation, hue,  p = 1.0)
        self.is_geometric = False

class _Rotation(K.augmentation.RandomRotation):
    def __init__(self, degress):
        super(_Rotation, self).__init__(degress, p = 1.0)
        self.is_geometric = True

class _Shear(K.augmentation.RandomAffine):
    def __init__(self, shear):
        super(_Shear, self).__init__( degrees = 0, shear = shear,  p = 1.0, align_corners=False)
        self.is_geometric = True


class _Scale(K.augmentation.RandomAffine):
    def __init__(self, scale):
        super(_Scale, self).__init__( degrees = 0, scale = scale,  p = 1.0)
        self.is_geometric = True


# notice that the defect location is given by engineerign knowledege as a mask

# add more gradient field transformations
