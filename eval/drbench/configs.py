"""The three retrieval configs.

Each maps a tool the agent can call -> the search channel that serves it.

    Text        search_text   -> text     256-word OCR passages, top-5
    Page        search_page   -> page     whole-page images, top-3
    Text_Fig    search_text   -> text     the same passages, top-5
                search_figure -> figure   cropped figure images, top-5

`Page` and `Text_Fig` send real images to the model, so they need a vision model.
`Text` never sends an image and works with any text model.
"""
CONFIGS = {
    "Text":     {"search_text": "text"},
    "Page":     {"search_page": "page"},
    "Text_Fig": {"search_text": "text", "search_figure": "figure"},
}

NEEDS_VISION = {"Page", "Text_Fig"}


def resolve(name):
    """Config name, case-insensitive (`page` -> `Page`); unknown names are returned unchanged."""
    return {k.lower(): k for k in CONFIGS}.get(name.lower(), name)
