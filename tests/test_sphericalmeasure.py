import math

import pytest
from scipy.integrate import quad
from firedrake import *
from echemfem import SphericalMeasure


RADIAL_RANGE = (.5, 2.)
POLAR_RANGE = (.2, 1.1)


@pytest.fixture(scope="module")
def shell():
    r0, r1 = RADIAL_RANGE
    theta0, theta1 = POLAR_RANGE
    mesh = RectangleMesh(2, 2, r1 - r0, theta1 - theta0,
                         quadrilateral=True)
    mesh.coordinates.dat.data[:, 0] += r0
    mesh.coordinates.dat.data[:, 1] += theta0
    r, theta = SpatialCoordinate(mesh)
    cells = FunctionSpace(mesh, "DQ", 0)
    inner_cells = Function(cells).interpolate(
        conditional(r < (r0 + r1) / 2, 1., 0.))
    outer_cells = Function(cells).interpolate(1 - inner_cells)
    facets = FunctionSpace(mesh, "HDiv Trace", 0)
    radial_faces = Function(facets).interpolate(
        conditional(abs(r - (r0 + r1) / 2) < 1e-8, 1., 0.))
    angular_faces = Function(facets).interpolate(
        conditional(abs(theta - (theta0 + theta1) / 2) < 1e-8, 1., 0.))
    mesh = RelabeledMesh(mesh, [inner_cells, outer_cells,
                               radial_faces, angular_faces], [1, 2, 11, 12])
    return mesh


def measure(mesh, integral_type):
    r, theta = SpatialCoordinate(mesh)
    return SphericalMeasure(r, theta, integral_type, domain=mesh,
                            metadata={"quadrature_degree": 24})


def angular_factor():
    theta0, theta1 = POLAR_RANGE
    return math.cos(theta0) - math.cos(theta1)


def test_volume(shell):
    r0, r1 = RADIAL_RANGE
    expected = (r1**3 - r0**3) / 3 * angular_factor()
    assert assemble(1 * measure(shell, "cell")) == pytest.approx(expected,
                                                               abs=1e-12)


def test_volume_subdomains(shell):
    r0, r1 = RADIAL_RANGE
    midpoint = (r0 + r1) / 2
    dx_s = measure(shell, "cell")
    inner = (midpoint**3 - r0**3) / 3 * angular_factor()
    full = (r1**3 - r0**3) / 3 * angular_factor()
    assert assemble(1 * dx_s(1)) == pytest.approx(inner, abs=1e-12)
    assert assemble(1 * dx_s((1, 2))) == pytest.approx(full, abs=1e-12)


@pytest.mark.parametrize("marker,radius", ((1, .5), (2, 2.)))
def test_radial_surface_area(shell, marker, radius):
    expected = radius**2 * angular_factor()
    assert assemble(1 * measure(shell, "exterior_facet")(marker)) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("marker,angle", ((3, .2), (4, 1.1)))
def test_angular_surface_area(shell, marker, angle):
    r0, r1 = RADIAL_RANGE
    expected = (r1**2 - r0**2) / 2 * math.sin(angle)
    assert assemble(1 * measure(shell, "exterior_facet")(marker)) == pytest.approx(expected, abs=1e-12)


def test_surface_subdomains(shell):
    r0, r1 = RADIAL_RANGE
    theta0, theta1 = POLAR_RANGE
    expected = ((r0**2 + r1**2) * angular_factor()
                + (r1**2 - r0**2) / 2
                * (math.sin(theta0) + math.sin(theta1)))
    ds_s = measure(shell, "exterior_facet")
    assert assemble(1 * ds_s((1, 2, 3, 4))) == pytest.approx(expected,
                                                          abs=1e-12)


def test_interior_surface_area(shell):
    r0, r1 = RADIAL_RANGE
    theta0, theta1 = POLAR_RANGE
    radial = ((r0 + r1) / 2)**2 * angular_factor()
    angular = (r1**2 - r0**2) / 2 * math.sin((theta0 + theta1) / 2)
    dS_s = measure(shell, "interior_facet")
    assert assemble(1 * dS_s(11)) == pytest.approx(radial, abs=1e-12)
    assert assemble(1 * dS_s(12)) == pytest.approx(angular, abs=1e-12)
    assert assemble(1 * dS_s((11, 12))) == pytest.approx(radial + angular,
                                                      abs=1e-12)
    assert assemble(1 * dS_s) == pytest.approx(radial + angular, abs=1e-12)


def test_oblique_surface_area():
    mesh = RectangleMesh(2, 2, 1., 1., quadrilateral=True)
    slope = .25
    coords = mesh.coordinates.dat.data
    coords[:, 1] += .3 + slope * coords[:, 0]
    coords[:, 0] += 1.
    # Revolving the meridian theta=.3+.25*(r-1) gives this surface area
    # per unit azimuth. Evaluate its independent line integral accurately.
    expected, _ = quad(lambda r: r * math.sin(.3 + slope * (r - 1))
                       * math.sqrt(1 + slope**2 * r**2), 1., 2.,
                       epsabs=1e-13, epsrel=1e-13)
    assert assemble(1 * measure(mesh, "exterior_facet")(3)) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("marker", (3, 4))
def test_polar_surface_area_is_zero(marker):
    mesh = RectangleMesh(2, 2, 1., math.pi, quadrilateral=True)
    mesh.coordinates.dat.data[:, 0] += 1.
    assert abs(assemble(1 * measure(mesh, "exterior_facet")(marker))) < 1e-12
