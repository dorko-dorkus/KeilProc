from .run_easy import SitePreset

PRESETS: dict[str, SitePreset] = {
    "DefaultSite": SitePreset(name="DefaultSite", geometry={}, instruments={}, defaults={}),
    # add real site presets here
}
