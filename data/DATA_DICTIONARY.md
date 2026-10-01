# Data Dictionary

CMS Medicare Physician & Other Practitioners - by Provider and Service (2024)

| Column | Description |
|---|---|
| `Rndrng_NPI` | National Provider Identifier. Unique ID for the provider rendering the service. ID only; do **not** feed into ML. |
| `Rndrng_Prvdr_Last_Org_Name` | Provider's last name if an individual, or organization name if an organization. |
| `Rndrng_Prvdr_First_Name` | Provider's first name if an individual; blank for an organization. |
| `Rndrng_Prvdr_MI` | Middle initial for individuals; blank for organizations. |
| `Rndrng_Prvdr_Crdntls` | Provider credentials, such as MD, DO, NP. |
| `Rndrng_Prvdr_Ent_Cd` | Whether the provider is an individual (`I`) or an organization (`O`). |
| `Rndrng_Prvdr_St1` | Street address line 1 from NPPES. |
| `Rndrng_Prvdr_St2` | Street address line 2. |
| `Rndrng_Prvdr_City` | Provider city. |
| `Rndrng_Prvdr_State_Abrvtn` | Provider state or territory abbreviation. |
| `Rndrng_Prvdr_State_FIPS` | Standard numeric code for the state or area. |
| `Rndrng_Prvdr_Zip5` | Provider's 5-digit ZIP code. |
| `Rndrng_Prvdr_RUCA` | Rural-Urban Commuting Area classification code. |
| `Rndrng_Prvdr_RUCA_Desc` | Human-readable RUCA description. |
| `Rndrng_Prvdr_Cntry` | Provider country code. |
| `Rndrng_Prvdr_Type` | Provider specialty/type, derived from the specialty associated with their largest number of services. |
| `Rndrng_Prvdr_Mdcr_Prtcptg_Ind` | Whether the provider participates in Medicare and/or accepts assignment of Medicare allowed amounts. |
| `HCPCS_Cd` | Code identifying the medical procedure, service, or product. |
| `HCPCS_Desc` | Plain-English description of the HCPCS code. |
| `HCPCS_Drug_Ind` | Whether the HCPCS service is listed as a Medicare Part B drug. |
| `Place_Of_Srvc` | `F` = facility; `O` = non-facility (office-like setting). |
| `Tot_Benes` | Number of distinct Medicare beneficiaries receiving this service for this NPI + HCPCS + place of service combination. |
| `Tot_Srvcs` | Number of services provided. |
| `Tot_Bene_Day_Srvcs` | Number of distinct beneficiary-per-day services; avoids repeated counting when one patient receives several units of the same service on one day. |
| `Avg_Sbmtd_Chrg` | Average amount the provider submitted/billed for the services. |
| `Avg_Mdcr_Alowd_Amt` | Average amount Medicare considers allowed, including Medicare payment plus patient deductible/coinsurance and applicable third-party amounts. |
| `Avg_Mdcr_Pymt_Amt` | Average amount Medicare itself paid after deductible/coinsurance. |
| `Avg_Mdcr_Stdzd_Amt` | Medicare payment after geographic payment differences are standardized away. |