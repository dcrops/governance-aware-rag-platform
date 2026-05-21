def get_domain_profile(profile_name: str | None) -> str:
    """
    Returns domain-specific extraction guidance
    for retrieval and aggregation workflows.
    """

    if profile_name == "chapter_detection":
        return (
            "Treat phrases such as "
            "'new men of the X Chapter', "
            "'welcoming the X Chapter', "
            "and 'new X Chapter' "
            "as evidence of new chapters. "
        )

    return ""