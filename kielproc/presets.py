from .run_easy import SitePreset

# In a real deployment these would be loaded from an external presets file.
PRESETS: dict[str, SitePreset] = {
    "DefaultSite": SitePreset(
        name="DefaultSite",
        geometry={"duct_diameter_m": 2.5, "ports": 8, "weighting": "equal"},
        instruments={"vp_unit": "Pa", "temp_unit": "C"},
        defaults={"fallback_baro_Pa": 101325},
    )
}
