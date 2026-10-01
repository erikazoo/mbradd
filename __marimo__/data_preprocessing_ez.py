{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "Hbol",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/json": "[\"text/plain:data\\\\raw\\\\mpos24.csv\", \"text/plain:Accept-Ranges: bytes\\nAccess-Control-Allow-Origin: *\\nContent-Type: text/csv\\nX-Acquia-Subscription-UUID: d15be109-6227-4efe-9952-9fefe48ff19c\\nX-AH-Environment: prod\\nX-Content-Type-Options: nosniff\\nX-Request-ID: v-c8e50c16-64e6-11f1-a165-e7ec7d0e41c6\\nLast-Modified: Mon, 11 May 2026 18:46:30 GMT\\nContent-Length: 3250282192\\nExpires: Tue, 22 Sep 2026 03:18:43 GMT\\nCache-Control: max-age=0, no-cache, no-store\\nPragma: no-cache\\nDate: Tue, 22 Sep 2026 03:18:43 GMT\\nConnection: close\\nStrict-Transport-Security: max-age=31536000 ; includeSubDomains ; preload\\nX-XSS-Protection: 1; mode=block\\nAkamai-GRN: 0.ada03b17.1790047123.8288d20\\n\\n\"]"
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import urllib.request\n",
    "from pathlib import Path\n",
    "\n",
    "url = \"https://data.cms.gov/sites/default/files/2026-05/b5ebab5a-f490-418a-9bce-4b9f31419356/PHY_R26_P05_V10_D24_Prov_Svc.csv\" # pulled from data.cms.gov/data.json \n",
    "out = Path(\"data/raw/mpos24.csv\")\n",
    "out.parent.mkdir(parents=True, exist_ok=True)\n",
    "urllib.request.urlretrieve(url, out)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "MJUe",
   "metadata": {},
   "outputs": [],
   "source": [
    "import marimo as mo, pandas as pd, numpy as np, sklearn\n",
    "path = \"data/raw/mpos24.csv\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "vblA",
   "metadata": {
    "marimo": {
     "config": {
      "hide_code": true
     },
     "md_prefix": "r"
    }
   },
   "source": [
    "Goal: Preprocess and visualize data, with features that make the dataset most workable for IsolationForest."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bkHC",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Rndrng_NPI', 'Rndrng_Prvdr_Last_Org_Name', 'Rndrng_Prvdr_First_Name', 'Rndrng_Prvdr_MI', 'Rndrng_Prvdr_Crdntls', 'Rndrng_Prvdr_Ent_Cd', 'Rndrng_Prvdr_St1', 'Rndrng_Prvdr_St2', 'Rndrng_Prvdr_City', 'Rndrng_Prvdr_State_Abrvtn', 'Rndrng_Prvdr_State_FIPS', 'Rndrng_Prvdr_Zip5', 'Rndrng_Prvdr_RUCA', 'Rndrng_Prvdr_RUCA_Desc', 'Rndrng_Prvdr_Cntry', 'Rndrng_Prvdr_Type', 'Rndrng_Prvdr_Mdcr_Prtcptg_Ind', 'HCPCS_Cd', 'HCPCS_Desc', 'HCPCS_Drug_Ind', 'Place_Of_Srvc', 'Tot_Benes', 'Tot_Srvcs', 'Tot_Bene_Day_Srvcs', 'Avg_Sbmtd_Chrg', 'Avg_Mdcr_Alowd_Amt', 'Avg_Mdcr_Pymt_Amt', 'Avg_Mdcr_Stdzd_Amt']\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table border=\"1\" class=\"dataframe\"><thead><tr style=\"text-align: right;\"><th></th><th>0</th></tr></thead><tbody><tr><th>Rndrng_NPI</th><td>int64</td></tr><tr><th>Rndrng_Prvdr_Last_Org_Name</th><td>str</td></tr><tr><th>Rndrng_Prvdr_First_Name</th><td>str</td></tr><tr><th>Rndrng_Prvdr_MI</th><td>str</td></tr><tr><th>Rndrng_Prvdr_Crdntls</th><td>str</td></tr><tr><th>...</th><td>...</td></tr><tr><th>Tot_Bene_Day_Srvcs</th><td>int64</td></tr><tr><th>Avg_Sbmtd_Chrg</th><td>float64</td></tr><tr><th>Avg_Mdcr_Alowd_Amt</th><td>float64</td></tr><tr><th>Avg_Mdcr_Pymt_Amt</th><td>float64</td></tr><tr><th>Avg_Mdcr_Stdzd_Amt</th><td>float64</td></tr></tbody></table><p>28 rows × 1 columns</p>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sample = pd.read_csv(path, nrows=1000) # confirm variables\n",
    "print(sample.columns.tolist())\n",
    "sample.dtypes # must check data types to know what to fix/downcast"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "lEQa",
   "metadata": {},
   "outputs": [],
   "source": [
    "usecols = ['Rndrng_Prvdr_Last_Org_Name', 'Rndrng_Prvdr_First_Name', 'Rndrng_Prvdr_Crdntls', 'Rndrng_Prvdr_Ent_Cd', 'Rndrng_Prvdr_City', 'Rndrng_Prvdr_State_Abrvtn', 'Rndrng_Prvdr_Type', 'Rndrng_Prvdr_Mdcr_Prtcptg_Ind', 'HCPCS_Cd', 'HCPCS_Desc', 'HCPCS_Drug_Ind',  'Place_Of_Srvc', 'Tot_Benes', 'Tot_Srvcs', 'Tot_Bene_Day_Srvcs', 'Avg_Sbmtd_Chrg', 'Avg_Mdcr_Alowd_Amt', 'Avg_Mdcr_Pymt_Amt',  'Avg_Mdcr_Stdzd_Amt'] # columns selected for use throughout\n",
    "model_input = ['Tot_Benes', 'Tot_Srvcs', 'Tot_Bene_Day_Srvcs', 'Avg_Sbmtd_Chrg', 'Avg_Mdcr_Alowd_Amt', 'Avg_Mdcr_Pymt_Amt',  'Avg_Mdcr_Stdzd_Amt'] # specific features for model input"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "PKri",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/json": "[9781673, 19]"
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "mpos24 = pd.read_csv(path, usecols=usecols)\n",
    "mpos24.shape # 19 features, 9781673 rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "Xref",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dtypes:\n",
      " Tot_Benes               int64\n",
      "Tot_Srvcs             float64\n",
      "Tot_Bene_Day_Srvcs      int64\n",
      "Avg_Sbmtd_Chrg        float64\n",
      "Avg_Mdcr_Alowd_Amt    float64\n",
      "Avg_Mdcr_Pymt_Amt     float64\n",
      "Avg_Mdcr_Stdzd_Amt    float64\n",
      "dtype: object\n",
      "\n",
      "missing per column:\n",
      " Tot_Benes             0\n",
      "Tot_Srvcs             0\n",
      "Tot_Bene_Day_Srvcs    0\n",
      "Avg_Sbmtd_Chrg        0\n",
      "Avg_Mdcr_Alowd_Amt    0\n",
      "Avg_Mdcr_Pymt_Amt     0\n",
      "Avg_Mdcr_Stdzd_Amt    0\n",
      "dtype: int64\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table border=\"1\" class=\"dataframe\"><thead><tr style=\"text-align: right;\"><th></th><th>Tot_Benes</th><th>Tot_Srvcs</th><th>Tot_Bene_Day_Srvcs</th><th>Avg_Sbmtd_Chrg</th><th>Avg_Mdcr_Alowd_Amt</th><th>Avg_Mdcr_Pymt_Amt</th><th>Avg_Mdcr_Stdzd_Amt</th></tr></thead><tbody><tr><th>count</th><td>9781673.0</td><td>9781673.0</td><td>9781673.0</td><td>9781673.0</td><td>9781673.0</td><td>9781673.0</td><td>9781673.0</td></tr><tr><th>mean</th><td>85.0</td><td>281.8</td><td>137.3</td><td>430.8</td><td>106.4</td><td>83.0</td><td>82.5</td></tr><tr><th>std</th><td>1312.2</td><td>6307.9</td><td>2181.4</td><td>1521.6</td><td>333.0</td><td>265.6</td><td>267.2</td></tr><tr><th>min</th><td>11.0</td><td>4.4</td><td>11.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><th>50%</th><td>32.0</td><td>43.0</td><td>41.0</td><td>180.0</td><td>71.6</td><td>52.5</td><td>52.7</td></tr><tr><th>90%</th><td>157.0</td><td>324.0</td><td>266.0</td><td>821.2</td><td>177.8</td><td>142.5</td><td>137.8</td></tr><tr><th>99%</th><td>601.0</td><td>2753.0</td><td>1180.0</td><td>4293.0</td><td>826.0</td><td>656.4</td><td>652.1</td></tr><tr><th>max</th><td>896762.0</td><td>6486401.0</td><td>1835261.0</td><td>100000.0</td><td>53965.4</td><td>42994.9</td><td>42994.9</td></tr></tbody></table>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def _(model_input, mpos24):\n",
    "    print(\"dtypes:\\n\", mpos24[model_input].dtypes)\n",
    "    print(\"\\nmissing per column:\\n\", mpos24[model_input].isna().sum())\n",
    "    return (usecols, model_input)\n",
    "\n",
    "_(model_input, mpos24)\n",
    "mpos24[model_input].describe(percentiles=[0.5, 0.9, 0.99]).round(1) # check skew"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3"
  },
  "marimo": {
   "header": "# /// script\n# requires-python = \">=3.14\"\n# dependencies = [\n#     \"marimo>=0.23.3\",\n#     \"pandas>=3.0.6\",\n#     \"scikit-learn>=1.9.1\",\n# ]\n# ///\n\n",
   "marimo_version": "0.24.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
