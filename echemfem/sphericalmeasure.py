from firedrake import FacetNormal
from ufl import sin, sqrt
from ufl.constantvalue import as_ufl
from ufl.domain import extract_domains

from echemfem.cylindricalmeasure import _WeightedMeasure


class SphericalMeasure(_WeightedMeasure):
    """Axisymmetric spherical integration, per unit azimuthal angle.

    The mesh coordinates are radius and polar angle. Cell and facet integrals
    use the physical volume and surface Jacobians respectively.
    """

    def __init__(self, radius, polar_angle, *args, **kwargs):
        self.radius = radius
        self.polar_angle = polar_angle
        super().__init__(*args, **kwargs)
        integral_type = self.integral_type()
        if integral_type == "cell":
            self.jacobian = radius**2 * sin(polar_angle)
        elif integral_type.startswith(("exterior_facet", "interior_facet")):
            domain = self.ufl_domain()
            if domain is None:
                domains = extract_domains(as_ufl(radius) + as_ufl(polar_angle))
                if len(domains) != 1:
                    raise ValueError("Spherical facet integration requires one mesh domain.")
                domain, = domains
            normal = FacetNormal(domain)
            if integral_type.startswith("interior_facet"):
                normal = normal("+")
            self.jacobian = radius * sin(polar_angle) * sqrt(
                radius**2 * normal[0]**2 + normal[1]**2)
        else:
            raise NotImplementedError(
                f"Spherical integration is not implemented for {integral_type} measures.")

    def reconstruct(self, integral_type=None, subdomain_id=None, domain=None,
                    metadata=None, subdomain_data=None, intersect_measures=None):
        return SphericalMeasure(
            self.radius, self.polar_angle,
            **self._measure_parameters(integral_type, subdomain_id, domain,
                                      metadata, subdomain_data, intersect_measures))
