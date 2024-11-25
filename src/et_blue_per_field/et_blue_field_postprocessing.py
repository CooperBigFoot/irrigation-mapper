import ee
from typing import Optional, Dict, Any


def create_nonzero_stats_reducer() -> ee.Reducer:
    """Create reducer for non-zero ET statistics."""
    return ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)


def compute_regional_stats(
    image: ee.Image,
    geometry: ee.Geometry,
    reducer: ee.Reducer,
    band_name: str,
    scale: float,
    max_pixels: int,
    projection: ee.Projection,
) -> ee.Dictionary:
    """Compute regional statistics for an image."""
    return ee.Dictionary(
        image.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=scale,
            maxPixels=max_pixels,
            crs=projection,
        )
    )


def compute_field_et_stats(
    et_image: ee.Image,
    fields: ee.FeatureCollection,
    et_band_name: str = "ET",
    scale: Optional[float] = None,
    max_pixels: int = int(1e9),
) -> ee.FeatureCollection:
    """
    Compute ET statistics for each field in a feature collection.

    Args:
        et_image: Input ET image
        fields: Feature collection of field boundaries
        et_band_name: Name of the ET band in the image
        scale: Scale in meters for computation. If None, uses native scale of the image
        max_pixels: Maximum number of pixels to process in reduction operations

    Returns:
        FeatureCollection with added properties:
        - median_et_blue: median ET value for each field
        - mean_et_nonzero: mean ET value excluding zero pixels
        - std_dev_et_nonzero: standard deviation of non-zero ET values
        - zero_fraction: fraction of pixels with 0 value in each field
    """
    projection = et_image.projection()
    if scale is None:
        scale = projection.nominalScale()

    et_image = et_image.setDefaultProjection(projection)

    def compute_feature_stats(feature: ee.Feature) -> ee.Feature:
        geometry = feature.geometry()

        # Create masks
        zero_mask = et_image.select(et_band_name).eq(0)
        nonzero_et = et_image.select(et_band_name).updateMask(zero_mask.Not())

        # Compute statistics
        nonzero_stats = compute_regional_stats(
            nonzero_et,
            geometry,
            create_nonzero_stats_reducer(),
            et_band_name,
            scale,
            max_pixels,
            projection,
        )

        median_stats = compute_regional_stats(
            et_image.select(et_band_name),
            geometry,
            ee.Reducer.median(),
            et_band_name,
            scale,
            max_pixels,
            projection,
        )

        zero_stats = compute_regional_stats(
            zero_mask,
            geometry,
            ee.Reducer.mean(),
            et_band_name,
            scale,
            max_pixels,
            projection,
        )

        # Extract and set properties
        return feature.set(
            {
                "median_et_blue": median_stats.get(et_band_name),
                "mean_et_nonzero": nonzero_stats.get(f"{et_band_name}_mean"),
                "std_dev_et_nonzero": nonzero_stats.get(f"{et_band_name}_stdDev"),
                "zero_fraction": zero_stats.get(et_band_name),
            }
        )

    return fields.map(compute_feature_stats)


def compute_et_volume(fields: ee.FeatureCollection) -> ee.FeatureCollection:
    """
    Compute ET volume in cubic meters for each field.

    Args:
        fields: FeatureCollection with median_et_nonzero property

    Returns:
        FeatureCollection with new et_blue_m3 property
    """

    def add_volume(feature: ee.Feature) -> ee.Feature:
        area = feature.geometry().area()
        et_mm = ee.Number(feature.get("median_et_blue"))
        et_volume = et_mm.multiply(area).divide(1000)

        return feature.set({"et_blue_m3": et_volume})

    return fields.map(add_volume)
