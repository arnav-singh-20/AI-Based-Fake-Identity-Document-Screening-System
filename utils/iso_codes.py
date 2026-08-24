"""
Minimal ISO 3166-1 alpha-3 country code set, used to sanity-check the
nationality field pulled from OCR / MRZ. Not exhaustive of every
territory code MRZ uses (e.g. 'UTO' for utopia in test docs, 'D' for
Germany historically) — extend as needed for your demo document set.
"""

ISO_ALPHA3_CODES = {
    "AFG", "ALB", "DZA", "AND", "AGO", "ARG", "ARM", "AUS", "AUT", "AZE",
    "BHS", "BHR", "BGD", "BRB", "BLR", "BEL", "BLZ", "BEN", "BTN", "BOL",
    "BIH", "BWA", "BRA", "BRN", "BGR", "BFA", "BDI", "KHM", "CMR", "CAN",
    "CPV", "CAF", "TCD", "CHL", "CHN", "COL", "COM", "COG", "COD", "CRI",
    "HRV", "CUB", "CYP", "CZE", "DNK", "DJI", "DMA", "DOM", "ECU", "EGY",
    "SLV", "GNQ", "ERI", "EST", "SWZ", "ETH", "FJI", "FIN", "FRA", "GAB",
    "GMB", "GEO", "DEU", "GHA", "GRC", "GRD", "GTM", "GIN", "GNB", "GUY",
    "HTI", "HND", "HUN", "ISL", "IND", "IDN", "IRN", "IRQ", "IRL", "ISR",
    "ITA", "JAM", "JPN", "JOR", "KAZ", "KEN", "KIR", "PRK", "KOR", "KWT",
    "KGZ", "LAO", "LVA", "LBN", "LSO", "LBR", "LBY", "LIE", "LTU", "LUX",
    "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MHL", "MRT", "MUS", "MEX",
    "FSM", "MDA", "MCO", "MNG", "MNE", "MAR", "MOZ", "MMR", "NAM", "NRU",
    "NPL", "NLD", "NZL", "NIC", "NER", "NGA", "MKD", "NOR", "OMN", "PAK",
    "PLW", "PAN", "PNG", "PRY", "PER", "PHL", "POL", "PRT", "QAT", "ROU",
    "RUS", "RWA", "KNA", "LCA", "VCT", "WSM", "SMR", "STP", "SAU", "SEN",
    "SRB", "SYC", "SLE", "SGP", "SVK", "SVN", "SLB", "SOM", "ZAF", "SSD",
    "ESP", "LKA", "SDN", "SUR", "SWE", "CHE", "SYR", "TJK", "TZA", "THA",
    "TLS", "TGO", "TON", "TTO", "TUN", "TUR", "TKM", "TUV", "UGA", "UKR",
    "ARE", "GBR", "USA", "URY", "UZB", "VUT", "VAT", "VEN", "VNM", "YEM",
    "ZMB", "ZWE",
    # Common MRZ special-purpose / historical codes seen in sample datasets
    "UTO", "D", "GBD", "GBN", "GBO", "GBP", "GBS",
}


def is_valid_country_code(code: str) -> bool:
    if not code:
        return False
    return code.strip().upper().replace("<", "") in ISO_ALPHA3_CODES
