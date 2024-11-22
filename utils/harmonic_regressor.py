import ee
from typing import List, Dict, Any, Optional
import math


def add_temporal_bands(collection: ee.ImageCollection) -> ee.ImageCollection:
    """Add temporal and constant bands to each image in the collection.

    Args:
        collection: The input image collection.

    Returns:
        ee.ImageCollection: Collection with added temporal and constant bands.
    """
    reference_date = ee.Date("1970-01-01")

    def _add_bands(image: ee.Image) -> ee.Image:
        # Get projection once and reuse
        projection = image.select([0]).projection()

        # Calculate years since reference date
        date = ee.Date(image.get("system:time_start"))
        years = date.difference(reference_date, "year")

        # Create bands with consistent projection
        time_band = ee.Image(years).float().rename("t").setDefaultProjection(projection)
        constant_band = (
            ee.Image.constant(1).rename("constant").setDefaultProjection(projection)
        )

        # Add both bands at once to reduce operations
        return image.addBands([time_band, constant_band])

    return collection.map(_add_bands)


class HarmonicRegressor:
    """A class for performing harmonic regression on Earth Engine image time series.

    This class implements harmonic regression for time series analysis, particularly
    useful for analyzing seasonal patterns in vegetation indices or other temporal data.
    """

    def __init__(
        self,
        omega: float = 1.5,
        max_harmonic_order: int = 2,
        band_to_harmonize: str = "NDVI",
        parallel_scale: int = 2,
    ):
        """Initialize the HarmonicRegressor.

        Args:
            omega: Angular frequency (default: 1.5)
            max_harmonic_order: Maximum number of harmonics to use (default: 2)
            band_to_harmonize: Name of the band to perform harmonization on (default: 'NDVI')
            parallel_scale: Scale for parallel processing (default: 2)
        """
        self.omega = omega
        self.max_harmonic_order = max_harmonic_order
        self.band_to_harmonize = band_to_harmonize
        self.parallel_scale = parallel_scale
        self._regression_coefficients = None
        self._fitted_data = None

        # Pre-compute harmonic component names
        self._harmonic_names = ["constant", "t"] + [
            f"{trig}{i}"
            for i in range(1, self.max_harmonic_order + 1)
            for trig in ["cos", "sin"]
        ]

        # Pre-compute omega values for each harmonic
        self._omega_values = [
            2 * i * self.omega * math.pi for i in range(1, self.max_harmonic_order + 1)
        ]

    @property
    def harmonic_component_names(self) -> List[str]:
        """Get the names of harmonic components.

        Returns:
            List[str]: List of harmonic component names.
        """
        return self._harmonic_names

    def fit(self, image_collection: ee.ImageCollection) -> "HarmonicRegressor":
        """Fit the harmonic regression model to the input image collection.

        Args:
            image_collection: Input image collection with temporal bands.

        Returns:
            HarmonicRegressor: The fitted model instance.

        Raises:
            TypeError: If image_collection is not an ee.ImageCollection.
            ValueError: If required bands are missing.
        """
        if not isinstance(image_collection, ee.ImageCollection):
            raise TypeError("image_collection must be an ee.ImageCollection")

        self._validate_input_collection(image_collection)

        harmonic_collection = self._prepare_harmonic_collection(image_collection)
        self._regression_coefficients = self._compute_regression_coefficients(
            harmonic_collection
        )
        self._fitted_data = self._compute_fitted_values(
            harmonic_collection, self._regression_coefficients
        )
        return self

    def predict(self, image_collection: ee.ImageCollection) -> ee.ImageCollection:
        """Predict using the fitted harmonic regression model.

        Args:
            image_collection: Input image collection for prediction.

        Returns:
            ee.ImageCollection: Collection with predicted values.

        Raises:
            ValueError: If the model hasn't been fitted.
        """
        if self._regression_coefficients is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        harmonic_collection = self._prepare_harmonic_collection(image_collection)
        return self._compute_fitted_values(
            harmonic_collection, self._regression_coefficients
        )

    def get_phase_amplitude(self) -> ee.Image:
        """Calculate phase and amplitude from regression coefficients.

        Returns:
            ee.Image: Image with phase and amplitude bands.

        Raises:
            ValueError: If the model hasn't been fitted.
        """
        if self._regression_coefficients is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._calculate_phase_amplitude()

    def _validate_input_collection(self, collection: ee.ImageCollection) -> None:
        """Validate the input collection has required bands.

        Args:
            collection: Input image collection to validate.

        Raises:
            ValueError: If required bands are missing.
        """
        first_image = collection.first()
        required_bands = ["t", self.band_to_harmonize]

        band_names = first_image.bandNames()
        missing_bands = [
            band for band in required_bands if not band_names.contains(band).getInfo()
        ]

        if missing_bands:
            raise ValueError(f"Missing required bands: {missing_bands}")

    def _prepare_harmonic_collection(
        self, image_collection: ee.ImageCollection
    ) -> ee.ImageCollection:
        """Prepare the input collection for harmonic regression.

        Args:
            image_collection: Input image collection.

        Returns:
            ee.ImageCollection: Collection with harmonic components.
        """
        return image_collection.map(self._add_harmonic_components)

    def _add_harmonic_components(self, image: ee.Image) -> ee.Image:
        """Add harmonic component bands to an image.

        Args:
            image: Input image.

        Returns:
            ee.Image: Image with added harmonic components.
        """
        time = image.select("t")
        harmonic_bands = []

        for i, omega_i in enumerate(self._omega_values, 1):
            time_radians = time.multiply(omega_i)
            harmonic_bands.extend(
                [
                    time_radians.cos().rename(f"cos{i}"),
                    time_radians.sin().rename(f"sin{i}"),
                ]
            )

        return image.addBands(harmonic_bands)

    def _compute_regression_coefficients(
        self, harmonic_collection: ee.ImageCollection
    ) -> ee.Image:
        """Compute regression coefficients.

        Args:
            harmonic_collection: Collection with harmonic components.

        Returns:
            ee.Image: Image containing regression coefficients.
        """
        regression_input_bands = ee.List(self._harmonic_names).add(
            self.band_to_harmonize
        )

        # Pre-select required bands
        selected_collection = harmonic_collection.select(regression_input_bands)

        regression_result = selected_collection.reduce(
            ee.Reducer.linearRegression(numX=len(self._harmonic_names), numY=1),
            parallelScale=self.parallel_scale,
        )

        return (
            regression_result.select("coefficients")
            .arrayProject([0])
            .arrayFlatten([self._harmonic_names])
        )

    def _compute_fitted_values(
        self, harmonic_collection: ee.ImageCollection, coefficients: ee.Image
    ) -> ee.ImageCollection:
        """Compute fitted values using regression coefficients.

        Args:
            harmonic_collection: Collection with harmonic components.
            coefficients: Regression coefficients.

        Returns:
            ee.ImageCollection: Collection with fitted values.
        """

        def compute_fitted(image: ee.Image) -> ee.Image:
            selected_image = image.select(self._harmonic_names)

            fitted_values = (
                selected_image.multiply(coefficients)
                .reduce(ee.Reducer.sum())
                .rename("fitted")
            )

            return image.addBands(fitted_values)

        return harmonic_collection.map(compute_fitted)

    def _calculate_phase_amplitude(self) -> ee.Image:
        """Calculate phase and amplitude from regression coefficients.

        Returns:
            ee.Image: Image with phase and amplitude bands.
        """
        components = []
        for i in range(1, self.max_harmonic_order + 1):
            coeff_pair = self._regression_coefficients.select([f"cos{i}", f"sin{i}"])

            # Calculate phase and amplitude together
            phase = coeff_pair.select(1).atan2(coeff_pair.select(0)).rename(f"phase{i}")
            amplitude = coeff_pair.reduce(ee.Reducer.hypot()).rename(f"amplitude{i}")

            components.extend([phase, amplitude])

        return ee.Image.cat(components)

    def get_coefficients(self) -> Optional[ee.Image]:
        """Get the regression coefficients.

        Returns:
            Optional[ee.Image]: Regression coefficients if model is fitted, None otherwise.
        """
        return self._regression_coefficients

    def get_fitted_data(self) -> Optional[ee.ImageCollection]:
        """Get the fitted data.

        Returns:
            Optional[ee.ImageCollection]: Fitted data if model is fitted, None otherwise.
        """
        return self._fitted_data
