from flask import Flask, jsonify, request
from flask_cors import CORS
from waitress import serve


# Numpy
import numpy as np

# Einsteinpy
import sympy
from sympy import symbols, sin, Function, diag
from einsteinpy.symbolic import (
    MetricTensor,
    RicciTensor,
    RicciScalar,
    RiemannCurvatureTensor,
)

from einsteinpy.symbolic.predefined import Kerr, KerrNewman, Schwarzschild


from Tensor import Tensor

print("Iniciando o servidor Flask na porta http://10.0.0.106:8081 ")

# Criando aplicação
app = Flask(__name__)
app.config['APPLICATION_ROOT'] = '/projeto2'  # Define o APPLICATION_ROOT
CORS(app, resources={r"/projeto2/*": {"origins": ["http://cloudhub.iprj.uerj.br", "https://cloudhub.iprj.uerj.br"]}})


# Rota Default
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Hello world"})

# Rota para métricas
@app.route("/metricas", methods=["GET"])
def get_metricas():
    metricas = [
        {"value": "Schwarzschild"},
        {"value": "Kerr"},
        {"value": "KerrNewman"},
        {"value": "FLRW"},
    ]
    return jsonify(metricas)

# Rota para cálculos de tensores
@app.route("/tensores", methods=["POST"])
def calcular_tensores():
    metrica = request.json["metrica"]
    tipo = request.json["tipo"]

    tensor = Tensor(metric=metrica)

    if tipo == "tensor":
        result = tensor.get_tensor()
    elif tipo == "riemann":
        result = tensor.get_riemann_tensor()
    elif tipo == "ricci":
        result = tensor.get_ricci_tensor()
    elif tipo == "ricciScalar":
        result = tensor.get_ricci_scalar()
    else:
        return jsonify(error="Tipo de tensor inválido")

    return jsonify(result=result)

# Classe Tensor
class Tensor:
    def __init__(self, metric="Schwarzschild"):
        self.__metric = self.__get_metric(metric)

    def __get_metric(self, metric_name):  # noqa: C901
        """Retorna uma metrica prédefinida usando o nome informado."""
        if metric_name == "FLRW":
            k = symbols("k")
            a = Function("a")
            syms = symbols("t  r theta phi")
            t, r, th, ph = syms
            m = diag(
                -1,
                (a(t) ** 2) / (1 - k * (r**2)),
                ((a(t) ** 2) * (r**2)),
                (((a(t) ** 2) * ((r * sin(th)) ** 2))),
            ).tolist()

            return MetricTensor(m, syms)
        elif metric_name == "Kerr":
            return Kerr()
        elif metric_name == "KerrNewman":
            return KerrNewman()
        elif metric_name == "Schwarzschild":
            return Schwarzschild()
        else:
            raise ValueError("Metrica não implementada.")

    def get_tensor(self):
        """Retorna o tensor da metrica."""
        return str(self.__metric.tensor())

    def get_ricci_scalar(self):
        """Retorna o escalar de Ricci."""
        return str(RicciScalar.from_metric(self.__metric).expr)

    def get_ricci_tensor(self):
        """Retorna o tensor de Ricci."""
        return str(RicciTensor.from_metric(self.__metric).tensor())

    def get_riemann_tensor(self):
        """Retorna o tensor de Riemann."""
        return str(RiemannCurvatureTensor.from_metric(self.__metric).tensor())

if __name__ == "__main__":
    # Iniciando o servidor com o APPLICATION_ROOT configurado
    serve(app, host="0.0.0.0", port=8081)
