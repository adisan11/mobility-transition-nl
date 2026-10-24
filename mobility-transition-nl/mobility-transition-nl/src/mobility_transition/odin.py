"""ODiN-specific helpers."""
import numpy as np
import pandas as pd


def clean_opid(series):
    """String OPID; float-read IDs like 12345.0 become '12345'; blanks become <NA>."""
    numeric = pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.strip()
    is_int = numeric.notna() & np.isclose(numeric, np.floor(numeric))
    cleaned.loc[is_int] = numeric.loc[is_int].astype("Int64").astype("string")
    return cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def household_car_status(cars, fuel):
    """
    No car = HHAuto 0; ICEV = HHAuto >= 1 and BrandstofPa1 in 1-3 (petrol, diesel, LPG); EV = HHAuto >= 1 and fuel 4.
    Anything else is 'Unclassified'. BrandstofPa1 describes the household's youngest car, so 'EV' means that car is
    electric, not necessarily that every household car is.
    """
    cars, fuel = pd.to_numeric(cars, errors="coerce"), pd.to_numeric(fuel, errors="coerce")
    owns = cars.ge(1)
    return pd.Series(np.select([owns & fuel.isin([1, 2, 3]), owns & fuel.eq(4), cars.eq(0)], ["ICEV", "EV", "No car"],
                               default="Unclassified"), index=cars.index)
