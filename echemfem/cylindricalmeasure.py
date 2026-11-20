from ufl.measure import Measure
from ufl.core.expr import Expr
from ufl.constantvalue import as_ufl


class _WeightedMeasure(Measure):
    """Apply a coordinate weight while leaving integral construction to UFL."""

    def __init__(self, *args, **kwargs):
        # UFL releases before intersection measures do not accept this keyword.
        if not hasattr(Measure, "intersect_measures"):
            intersections = kwargs.pop("intersect_measures", None)
            if intersections:
                raise NotImplementedError("This UFL version does not support intersection measures.")
        super().__init__(*args, **kwargs)

    def _measure_parameters(self, integral_type=None, subdomain_id=None,
                            domain=None, metadata=None, subdomain_data=None,
                            intersect_measures=None):
        parameters = {
            "integral_type": self.integral_type() if integral_type is None else integral_type,
            "domain": self.ufl_domain() if domain is None else domain,
            "subdomain_id": self.subdomain_id() if subdomain_id is None else subdomain_id,
            "metadata": self.metadata() if metadata is None else metadata,
            "subdomain_data": self.subdomain_data() if subdomain_data is None else subdomain_data,
        }
        if hasattr(Measure, "intersect_measures"):
            parameters["intersect_measures"] = (
                self.intersect_measures() if intersect_measures is None else intersect_measures)
        elif intersect_measures:
            raise NotImplementedError("This UFL version does not support intersection measures.")
        return parameters

    def __rmul__(self, integrand):
        if isinstance(integrand, (int, float)):
            integrand = as_ufl(integrand)
        if not isinstance(integrand, Expr):
            return NotImplemented
        # A plain measure also avoids applying the weight twice when UFL splits
        # a tuple of subdomain IDs into separate integrals. Its domain inference
        # sees the coordinate weight, so literal integrands such as 1*dx work.
        measure = Measure(**self._measure_parameters())
        return measure.__rmul__(self.jacobian * integrand)


class CylindricalMeasure(_WeightedMeasure):
    """Axisymmetric cylindrical integration, per unit azimuthal angle."""

    def __init__(self, radius, *args, **kwargs):
        self.radius = radius
        self.jacobian = radius
        super().__init__(*args, **kwargs)

    def reconstruct(self, integral_type=None, subdomain_id=None, domain=None,
                    metadata=None, subdomain_data=None, intersect_measures=None):
        return CylindricalMeasure(
            self.radius,
            **self._measure_parameters(integral_type, subdomain_id, domain,
                                      metadata, subdomain_data, intersect_measures))
