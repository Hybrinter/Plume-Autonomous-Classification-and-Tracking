# Industrial-area inventory (design study)

Raw downloads stay out of git. Re-fetch:

```text
# Climate TRACE v5.10.0 CO2 sector packages (CC BY 4.0)
# https://downloads.climatetrace.org/latest/sector_packages/co2/
power.zip, manufacturing.zip, fossil_fuel_operations.zip

# WRI Global Power Plant Database v1.3 (CC BY 4.0)
https://wri-dataportal-prod.s3.amazonaws.com/manual/global_power_plant_database_v_1_3.zip
```

Unpack under `raw/ct_power`, `raw/ct_manufacturing`, `raw/ct_fossil`, and
`raw/gppd`. GEM operating plants are pulled at run time from
`https://api.globalenergymonitor.org/assets`.
