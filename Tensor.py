from einsteinpy.symbolic import (
    MetricTensor,
    RicciScalar,
    RicciTensor,
    RiemannCurvatureTensor,
)
from einsteinpy.symbolic.predefined import (
    AlcubierreWarp,
    AntiDeSitter,
    AntiDeSitterStatic,
    BarriolaVilekin,
    BertottiKasner,
    BesselGravitationalWave,
    CMetric,
    Davidson,
    DeSitter,
    Ernst,
    Godel,
    JanisNewmanWinicour,
    Kerr,
    KerrNewman,
    Minkowski,
    MinkowskiCartesian,
    MinkowskiPolar,
    ReissnerNordstorm,
    Schwarzschild,
)
from sympy import Function, diag, sin, symbols


class Tensor:
    def __init__(self, metric="Schwarzschild"):
        self.__metric = self.__get_metric(metric)

    def __get_metric(self, metric_name):  # noqa: C901
        """Retorna uma metrica prédefinida usando o nome informado."""
        if metric_name == "AlcubierreWarp":
            return AlcubierreWarp()
        elif metric_name == "AntiDeSitter":
            return AntiDeSitter()
        elif metric_name == "AntiDeSitterStatic":
            return AntiDeSitterStatic()
        elif metric_name == "BarriolaVilekin":
            return BarriolaVilekin()
        elif metric_name == "BertottiKasner":
            return BertottiKasner()
        elif metric_name == "BesselGravitationalWave":
            return BesselGravitationalWave()
        elif metric_name == "CMetric":
            return CMetric()
        elif metric_name == "Davidson":
            return Davidson()
        elif metric_name == "DeSitter":
            return DeSitter()
        elif metric_name == "Ernst":
            return Ernst()
        elif metric_name == "FLRW":
            # Definir a constante k
            k = symbols("k")

            a = Function("a")
            # Definir as variaveis das metricas
            syms = symbols("t  r theta phi")
            t, r, th, ph = syms

            # Definir o tensor da métrica
            m = diag(
                -1,
                (a(t) ** 2) / (1 - k * (r**2)),
                ((a(t) ** 2) * (r**2)),
                (((a(t) ** 2) * ((r * sin(th)) ** 2))),
            ).tolist()

            return MetricTensor(m, syms)
        elif metric_name == "Godel":
            return Godel()
        elif metric_name == "JanisNewmanWinicour":
            return JanisNewmanWinicour()
        elif metric_name == "Kerr":
            return Kerr()
        elif metric_name == "KerrNewman":
            return KerrNewman()
        elif metric_name == "Minkowski":
            return Minkowski()
        elif metric_name == "MinkowskiCartesian":
            return MinkowskiCartesian()
        elif metric_name == "MinkowskiPolar":
            return MinkowskiPolar()
        elif metric_name == "ReissnerNordstorm":
            return ReissnerNordstorm()
        elif metric_name == "Schwarzschild":
            return Schwarzschild()
        else:
            raise ValueError("Metrica não implementada.")

    def get_tensor(self):
        """Retorna o tensor da metrica.

        Resposta para Schwarzschild:
            [[1 - r_s/r, 0, 0, 0], [0, -1/(c**2*(1 - r_s/r)), 0, 0], [0, 0, -r**2/c**2, 0], [0, 0, 0, -r**2*sin(theta)**2/c**2]]
        """
        return self.__metric.tensor()

    def get_ricci_scalar(self):
        """Retorna o escalar de Ricci.

        Resposta para Schwarzschild:
            0
        """
        return RicciScalar.from_metric(self.__metric).expr

    def get_ricci_tensor(self):
        """Retorna o tensor de Ricci.

        Resposta para Schwarzschild:
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        """
        return RicciTensor.from_metric(self.__metric).tensor()

    def get_riemann_tensor(self):
        """Retorna o tensor de Riemann.

        Resposta para Schwarzschild:
            [[[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, r_s/(r**2*(r - r_s)), 0, 0], [-r_s/(r**2*(r - r_s)), 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, -r_s/(2*r), 0], [0, 0, 0, 0], [r_s/(2*r), 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, -r_s*sin(theta)**2/(2*r)], [0, 0, 0, 0], [0, 0, 0, 0], [r_s*sin(theta)**2/(2*r), 0, 0, 0]]], [[[0, r_s*c**2*(r - r_s)/r**4, 0, 0], [r_s*c**2*(-r + r_s)/r**4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, -r_s/(2*r), 0], [0, r_s/(2*r), 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, -r_s*sin(theta)**2/(2*r)], [0, 0, 0, 0], [0, r_s*sin(theta)**2/(2*r), 0, 0]]], [[[0, 0, r_s*c**2*(-r + r_s)/(2*r**4), 0], [0, 0, 0, 0], [r_s*c**2*(r - r_s)/(2*r**4), 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, r_s/(2*r**2*(r - r_s)), 0], [0, -r_s/(2*r**2*(r - r_s)), 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, r_s*sin(theta)**2/r], [0, 0, -r_s*sin(theta)**2/r, 0]]], [[[0, 0, 0, r_s*c**2*(-r + r_s)/(2*r**4)], [0, 0, 0, 0], [0, 0, 0, 0], [r_s*c**2*(r - r_s)/(2*r**4), 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, r_s/(2*r**2*(r - r_s))], [0, 0, 0, 0], [0, -r_s/(2*r**2*(r - r_s)), 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -r_s/r], [0, 0, r_s/r, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]]
        """
        return RiemannCurvatureTensor.from_metric(self.__metric).tensor()


# if __name__ == '__main__':
#     tensor = Tensor('KerrNewman')

#     print(tensor.get_tensor())
#     print(tensor.get_ricci_scalar())
#     print(tensor.get_ricci_tensor())
#     print(tensor.get_riemann_tensor())
