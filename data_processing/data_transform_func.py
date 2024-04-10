
class DataScaler:
    def __init__(self):
        pass

    def rescale_vector(self, proprio_vector):
        return [self.rescale(jointValue) for jointValue in proprio_vector]

    def scale_vector_back(self, proprio_vector):
        return [self.scale_back(jointValue) for jointValue in proprio_vector]

    def rescale(self, num, joint=None):
        return num / 100

    def scale_back(self, num, joint=None):
        return num * 100


class SupremumInfinumScaler(DataScaler):
    def __init__(self):
        super().__init__()
        self.absoluteMin = -95
        self.absoluteMax = 270

    def rescale_vector(self, proprio_vector):
        return [self.rescale(jointValue) for jointValue in proprio_vector]

    def scale_vector_back(self, proprio_vector):
        return [self.scale_back(jointValue) for jointValue in proprio_vector]

    def rescale(self, num, joint=None):
        return (num - self.absoluteMin) / (self.absoluteMax - self.absoluteMin)

    def scale_back(self, num, joint=None):
        return (num*(self.absoluteMax - self.absoluteMin)) + self.absoluteMin

class LocalJointScaler(DataScaler):

    def __init__(self):
        super().__init__()
        self.joints_ranges = [
            (-95, 10), (-2, 161),
            (-37, 80), (15, 106),
            (-90, 90), (-91, 1),
            (-20, 40), (-2, 60),
            (8, 90), (-2, 92),
            (-2, 180), (-2, 90),
            (-2, 180), (-2, 90),
            (-2, 180), (-2, 270),
        ]

    def rescale_vector(self, proprio_vector):
        res = []
        for i in range(len(self.joints_ranges)):
            rescaled = self.rescale(proprio_vector[i], joint=self.joints_ranges[i])
            res.append(rescaled)
        return res
        #[self.rescale(proprio_vector[i], self.joints_ranges[i]) for i in range(len(self.joints_ranges))]

    def scale_vector_back(self, proprio_vector):
        return [self.scale_back(proprio_vector[i], self.joints_ranges[i]) for i in range(len(self.joints_ranges))]

    def scale_vector_back(self, proprio_vector, min, max):
        return [self.scale_back(proprio_vector[i], self.joints_ranges[i]) for i in range(min, max)]

    def rescale(self, num, joint):
        min_value, max_value = joint
        return (num - min_value) / (max_value - min_value)

    def scale_back(self, num, joint):
        min_value, max_value = joint
        return (num*(max_value - min_value)) + min_value

