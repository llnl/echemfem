"""Coordinate measures must preserve UFL's form and reconstruction semantics."""

import pytest
from math import cos, sin
from firedrake import ExtrudedMesh, IntervalMesh, SpatialCoordinate, UnitSquareMesh, as_vector, assemble
from ufl.measure import Measure
from ufl.domain import extract_unique_domain

from echemfem import CylindricalMeasure, SphericalMeasure


@pytest.fixture(params=["cylindrical", "spherical"])
def coordinate_measure(request):
    mesh = UnitSquareMesh(1, 1)
    r, theta = SpatialCoordinate(mesh)
    if request.param == "cylindrical":
        return CylindricalMeasure(r, "cell")
    return SphericalMeasure(r, theta, "cell")


def test_literal_integrand_infers_coordinate_domain(coordinate_measure):
    integral, = (1 * coordinate_measure).integrals()
    assert integral.ufl_domain() == extract_unique_domain(coordinate_measure.radius)
    assert integral.integrand() == coordinate_measure.jacobian


def test_subdomains_apply_weight_once(coordinate_measure):
    configured = coordinate_measure((1, 2), degree=7)
    integrals = (1 * configured).integrals()
    assert {integral.subdomain_id() for integral in integrals} == {1, 2}
    for integral in integrals:
        assert integral.integrand() == coordinate_measure.jacobian
        assert integral.metadata() == {"quadrature_degree": 7}


def test_reconstruct_preserves_configuration(coordinate_measure):
    mesh = extract_unique_domain(coordinate_measure.radius)
    markers = object()
    configured = coordinate_measure.reconstruct(
        integral_type="exterior_facet", domain=mesh, subdomain_id=3,
        metadata={"quadrature_degree": 6}, subdomain_data=markers)
    copied = configured.reconstruct()
    assert type(copied) is type(coordinate_measure)
    assert copied.integral_type() == "exterior_facet"
    assert copied.ufl_domain() == mesh
    assert copied.subdomain_id() == 3
    assert copied.metadata() == {"quadrature_degree": 6}
    assert copied.subdomain_data() is markers
    assert copied.radius is coordinate_measure.radius
    if isinstance(copied, SphericalMeasure):
        assert copied.polar_angle is coordinate_measure.polar_angle


def test_reconstruct_preserves_intersection_domains(coordinate_measure):
    if not hasattr(Measure, "intersect_measures"):
        pytest.skip("UFL does not support intersection measures")
    other_mesh = UnitSquareMesh(1, 1)
    intersection = Measure("exterior_facet", domain=other_mesh)
    configured = coordinate_measure.reconstruct(
        intersect_measures=(intersection,))
    copied = configured.reconstruct(subdomain_id=(1, 2), metadata={"quadrature_degree": 4})
    assert copied.intersect_measures() == (intersection,)
    for integral in (1 * copied).integrals():
        assert integral.extra_domain_integral_type_map() == {
            intersection.ufl_domain(): "exterior_facet"}


def test_tensor_integrands_are_rejected(coordinate_measure):
    with pytest.raises(ValueError, match="Can only integrate scalar expressions"):
        as_vector((1, 2)) * coordinate_measure


@pytest.mark.parametrize("coordinates", ["cylindrical", "spherical"])
def test_extruded_coordinate_facet_reconstruction(coordinates):
    mesh = ExtrudedMesh(IntervalMesh(2, 1), layers=2, layer_height=0.5)
    mesh.coordinates.dat.data[:, 0] += 1
    mesh.coordinates.dat.data[:, 1] += 0.3
    r, theta = SpatialCoordinate(mesh)
    if coordinates == "cylindrical":
        measure = CylindricalMeasure(r, "cell", domain=mesh)
        expected = {
            "exterior_facet_vert": 3,
            "exterior_facet_top": 1.5,
            "exterior_facet_bottom": 1.5,
            "interior_facet_vert": 1.5,
            "interior_facet_horiz": 1.5,
        }
    else:
        measure = SphericalMeasure(r, theta, "cell", domain=mesh)
        angular_integral = cos(0.3) - cos(1.3)
        expected = {
            "exterior_facet_vert": 5 * angular_integral,
            "exterior_facet_top": 1.5 * sin(1.3),
            "exterior_facet_bottom": 1.5 * sin(0.3),
            "interior_facet_vert": 2.25 * angular_integral,
            "interior_facet_horiz": 1.5 * sin(0.8),
        }
    for integral_type, exact_area in expected.items():
        facet_measure = measure.reconstruct(integral_type=integral_type)
        area = assemble(1 * facet_measure(degree=20))
        assert area == pytest.approx(exact_area, rel=1e-12, abs=1e-12)
