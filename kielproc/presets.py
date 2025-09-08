from .run_easy import SitePreset

# In a real deployment these would be loaded from an external presets file.
PRESETS: dict[str, SitePreset] = {
    "DefaultSite": SitePreset(
        name="DefaultSite",
        geometry={"duct_width_m": 2.5, "duct_height_m": 2.5},
        instruments={"vp_unit": "Pa", "temp_unit": "C"},
        defaults={"fallback_baro_Pa": 101325},
    )
}
