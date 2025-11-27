
# Run if you need to install libraries required

"""
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn

"""

import pandas as pd                 # For Data Manipulation
import numpy as np                  # For Data Manipulation
import matplotlib.pyplot as plt     # For Data Visualisation
import seaborn as sns               # For Data Visualisation
import sqlite3                      # For SQL
import os                           # For Directory Management

most_popular_cars = ["volkswagen", "ford", "bmw", "audi", "mercedes",
                        "toyota","nissan", "kia", "hyundai", "peugeot"]

# loadTransform function loads the data, and is also able to convert specified columns to python string

def loadTransform(category, object_columns=None):


  try:
    script_dir = os.path.dirname(os.path.abspath(__file__)) # File path where script is running
    csv_path = os.path.join(script_dir, "csv-last-5-years", f"dft-road-casualty-statistics-{category}-last-5-years.csv")

  except:
    # __file__ does not exist in colab
    csv_path = f"./csv-last-5-years/dft-road-casualty-statistics-{category}-last-5-years.csv"


  df = pd.read_csv(csv_path)

  # Converts columns specified in parameters to python string (intended for columns of mixed types)

  if object_columns:
      for column in object_columns:
        if column == "date":
          df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        elif column == "time":
          df["time"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.time
        else:
            df[column] = df[column].astype(str)

  return df

# loadDescribe function: loads and describes the data

def fullDescribe(df):


  # (a) Dimension data

  print("NUMBER OF ROWS AND COLUMN: \n")
  print(df.shape) # Number of rows and columns
  print("\n")

  # (b) Attribute types

  print("COLUMNS AND DATA TYPES: \n");
  print(df.info()) # Columns and data types
  print("\n")

  # (c) Missing values

  # Making a dataframe of the columns and missing value/percentage

  print ("DATAFRAME OF MISSING: \n")
  df_replaceNA = df.replace([-1,"-1"],np.nan)
  dfMissing = pd.DataFrame({
    'Column Name': df_replaceNA.columns,
    'Missing Count': df_replaceNA.isnull().sum(),
    'Missing Percentage': (df_replaceNA.isnull().sum() / len(df_replaceNA) * 100).round(4)
    })
  print(dfMissing)
  print("\n")

  # (d) Descriptive statistics

  print("DESCRIPTIVE STATISTICS: \n")
  print(df_replaceNA.describe(include='all')) # Descriptive Statistics
  print("\n")

  return

# Function used for mapping brand in visualisation and data cleaning

def map_brand(model):
  model = str(model).lower()
  for brand in most_popular_cars:
      if brand in model:
          return brand.capitalize()
  return np.nan

# Visualisations for Casualty

def vizCasualty(df):

  df = df.copy()

  columns = ["age_band_of_casualty", "casualty_severity", "casualty_distance_banding"] # List of columns I will use for visualisations
  df = df[columns].copy()

  plt.figure(figsize=(12,6))

  # Plot 1

  plt.subplot(1,2,1)

   # Decoding the columns

  age_labels = {-1:"NA", 1:"0-5",2:"6-10",3:"11-15",4:"16-20",5:"21-25", 6:"26-35",
                7:"36-45",8:"46-55", 9:"56-65", 10:"66-75", 11:"Over 75"}


  age_order = ["0-5","6-10","11-15","16-20","21-25","26-35", "36-45","46-55",
               "56-65","66-75","Over 75"]

  severity_labels = {1:"Fatal", 2:"Serious", 3:"Slight"}

   # Adding decoded columns to DF (Mapping)

  df["age_group"] = df["age_band_of_casualty"].map(age_labels)
  df["age_group"] = pd.Categorical(df["age_group"], categories=age_order, ordered=True)
  df["severity_label"] = df["casualty_severity"].map(severity_labels)

  df_plot1 = df[df['age_group'] != "NA"].copy()

  # Preparing for use in stacked bar

  grouped = df_plot1.groupby(['age_group', "severity_label"], observed=False)
  grouped_counts = grouped.size()
  age_sev_count = grouped_counts.unstack()


  age_sev_count.plot(
      kind="bar",
      stacked=True,
      colormap="Set1", # Changing colour palette
      ax=plt.gca() # Making sure it plots in this subplot
    )


  plt.title("Number of Casualties by Age Group")
  plt.xlabel("Age Group")
  plt.ylabel("Number of Casualties")
  plt.xticks(rotation=45)
  plt.legend(title="Severity")


  # Plot 2

  plt.subplot(1,2,2)

  # Decoding the columns

  distance_labels = {-1:"NA", 1:"0-5", 2:"5-10", 3:"10-20", 4:"20-100", 5:"NA"} # Made 100km+ NA so I can remove later
  distance_order = ["0-5","5-10","10-20", "20-100"]
  bin_width = {"0-5":5,"5-10":5,"10-20":10, "20-100":80}

  # Adding decoded columns to DF (Mapping)

  df["distance_group"] = df["casualty_distance_banding"].map(distance_labels)
  df = df[df['distance_group'] != "NA"].copy()
  df["distance_group"] = pd.Categorical(df["distance_group"], categories=distance_order, ordered=True)


  # Adding frequency density

  df = df.groupby(["distance_group", "severity_label"], observed=False).size().reset_index(name="count")
  df["distance_bin_width"] = df["distance_group"].map(bin_width)
  df["log_freq_density"] = np.log((df["count"] / df["distance_bin_width"]) +1)

  df_pivot = df.pivot(index="distance_group", columns="severity_label", values="log_freq_density").fillna(0)

  # Plotting

  sns.heatmap(df_pivot, cmap="YlGnBu")
  plt.title("Heatmap of Log(FD) by Distance & Severity")
  plt.xlabel("Severity Level")
  plt.ylabel("Distance from Home Postcode (km)")
  plt.yticks(rotation=0)
  plt.tight_layout()
  plt.show()

  return

# Visualisations for Collision

def vizCollision(df):

  df = df.copy()

  columns = ["local_authority_district", "day_of_week", "speed_limit"]
  df = df[columns].copy()

  plt.figure(figsize=(12,6))

  # Plot 1

  # Decoding columns

  london_boroughs = {
    1: "Westminster", 2: "Camden", 3: "Islington", 4: "Hackney",
    5: "Tower Hamlets", 6: "Greenwich", 7: "Lewisham",
    8: "Southwark", 9: "Lambeth", 10: "Wandsworth", 11: "Hammersmith and Fulham",
    12: "Kensington and Chelsea", 13: "Waltham Forest", 14: "Redbridge",
    15: "Havering", 16: "Barking and Dagenham", 17: "Newham",
    18: "Bexley", 19: "Bromley", 20: "Croydon", 21: "Sutton",
    22: "Merton", 23: "Kingston upon Thames", 24: "Richmond upon Thames",
    25: "Hounslow", 26: "Hillingdon", 27: "Ealing", 28: "Brent", 29: "Harrow",
    30: "Barnet", 31: "Haringey", 32: "Enfield"}

  inner_keys = list(range(1, 13))

  df_plot1 = df[df["local_authority_district"].isin(london_boroughs.keys())].copy() # Df of London
  df_plot1["borough"] = df_plot1["local_authority_district"].map(london_boroughs)

  df_plot1["region"] = df_plot1["local_authority_district"].apply(
    lambda x: "Inner London" if x in inner_keys else "Outer London") # Mapping Region


  day_labels = {1:"Sunday", 2:"Monday", 3:"Tuesday", 4:"Wednesday",
                5:"Thursday", 6:"Friday", 7:"Saturday"}

  day_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


  df_plot1["day_label"] = df_plot1["day_of_week"].map(day_labels)
  df_plot1["day_label"] = pd.Categorical(df_plot1["day_label"], categories=day_order, ordered=True) # Mapping days (ordered)

  # Preparing for graph

  df_plot1 = df_plot1.groupby(["region", "day_label"],observed=False).size().reset_index(name="count") # Observed = false to silence warnings about future updates (Just doing want the terminal was saying)

  # Using proportions for clarity between regions

  region_totals = df_plot1.groupby('region')['count'].transform('sum')
  df_plot1['proportion'] = df_plot1['count'] / region_totals

  df_plot1 = df_plot1.pivot(index="day_label", columns="region", values="proportion")

  reordered = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  df_plot1 = df_plot1.reindex(reordered) # Reordering for graph

  # Plotting

  plt.subplot(1,2,1)

  df_plot1.plot(marker='o',
                ax=plt.gca()
  )
  plt.legend(title="Region")
  plt.title("Proportion of Collisions in London per Day by Region")
  plt.xlabel("Day of Week")
  plt.xticks(rotation=45)
  plt.ylabel("Proportion")

  # Plot 2


  df["speed_limit"]= df["speed_limit"].replace([-1,99],np.nan)
  df = df.dropna()
  df['speed_limit'] = df['speed_limit'].astype(int) # Because gets converted to float after NAs
  df = df.groupby("speed_limit").size().reset_index(name="count")


  # Plotting

  plt.subplot(1,2,2)


  wedges, texts, autotexts = plt.pie(
    df['count'],
    autopct='%1.1f%%')


  plt.title("Pie Chart of Collisions")
  plt.legend(wedges, df['speed_limit'], title="Speed Limit(Mph)",
             bbox_to_anchor=(1,1))

  plt.tight_layout()
  plt.show()

  return

# Visualisations for Vehicle

def vizVehicle(df):

  df = df.copy()

  columns = ["first_point_of_impact", "generic_make_model", "age_of_driver"]
  df = df[columns].copy()

  plt.figure(figsize=(12,6))

  # Plot 1

  impact_labels =  {1: "Front",2: "Back",3: "Offside",4: "Nearside",}


  # Mapping impact labels

  df_plot1 = df[df["first_point_of_impact"].isin(impact_labels.keys())].copy()
  df_plot1["impact_point"] = df_plot1["first_point_of_impact"].map(impact_labels)
  df_plot1 = df_plot1["impact_point"].value_counts().reindex(impact_labels.values())

  # Plotting

  plt.subplot(1,2,1)

  df_plot1.plot(kind="bar",
      ax=plt.gca(),
    )

  plt.title("Bar Chart of Collisions by Impact Point")
  plt.xlabel("Impact Point")
  plt.ylabel("Number of Collisions")
  plt.xticks(rotation=90)

  # Plot 2


  df["age_of_driver"]= df["age_of_driver"].replace(-1,np.nan)
  df = df.dropna()

  df["brand"] = df["generic_make_model"].apply(map_brand) # Mapping car brand using created function

  df = df.groupby("brand").agg(
    average_age = ("age_of_driver", "mean"),
    count =("brand", "size")).reset_index()

  total_drivers = df['count'].sum()
  df['proportion'] = df['count'] / total_drivers




  plt.subplot(1,2,2)


  sns.scatterplot(data=df,
                  x="average_age",
                  y="proportion",
                  hue="brand")

  plt.title("Proportion of Collisions by Top 10 Car Brands in the UK")
  plt.legend(title="Most Driven in UK")
  plt.xlabel("Average age")
  plt.ylabel("Proportion of collisions")
  plt.tight_layout()
  plt.show()

  return

def processCasualty(df):

  df = df.copy()

  # Calculating IQR

  Q1 = df["age_of_casualty"].quantile(0.25)
  Q3 = df["age_of_casualty"].quantile(0.75)
  IQR = Q3 - Q1

  lower = Q1 - (1.5 * IQR)
  upper = Q3 + (1.5 * IQR)

  df["age_of_casualty"] = df["age_of_casualty"].clip(lower, upper).astype(int) # Clip converts to float

  df = df.replace([-1,"-1"], np.nan) # Replacing -1 to accurately count what data is missing

  columns = df.columns[(df.isnull().sum() / len(df) * 100) < 3].tolist()

  df = df.dropna(subset=columns) # Dropping missing values in that column

  df["sex_of_casualty"] = df["sex_of_casualty"].replace(9,np.nan)
  df = df.dropna(subset=["sex_of_casualty"])

  df["is_male"] = (df["sex_of_casualty"] == 1).astype(int) # Converts True to 1 else 0


  # Missing values are code as -1, changing to na

  df= df.replace([-1,"-1"], np.nan)

  return df

def processCollision(df):

  df = df.copy()

  historic = [column for column in df.columns if "historic" in column] # Creates a list of columns that are historic
  columns = historic + ["enhanced_severity_collision", "collision_year", "local_authority_district"]
  df = df.drop(columns=columns)

  df["casualty_per_vehicle"] = df["number_of_casualties"] / df["number_of_vehicles"]

  normal_conditions = (
    (df["light_conditions"] == 1) &           # Daylight
    (df["weather_conditions"] == 1) &         # Fine no high winds
    (df["road_surface_conditions"] == 1) &    # Dry
    (df["special_conditions_at_site"] == 0))  # No special conditions

  df["normal_conditions"] = normal_conditions.astype(int) # Normal conditions 1 (True) else 0 (False)


  # Missing values are code as -1, changing to na

  df= df.replace([-1,"-1"], np.nan)

  return df

def processVehicle(df, most_popular_cars):

  df = df.copy()

  df["brand"] = df["generic_make_model"].apply(map_brand) # Mapping car brand using created function
  most_popular_cars = [brand.capitalize() for brand in most_popular_cars] # List needs to be captialized to match accurately
  df = df[df["brand"].isin(most_popular_cars)].copy()

  df["brand_code"] = df["brand"].astype("category").cat.codes + 1 # Creating brand codes starting from 1

  median = df["age_of_vehicle"].median()
  df["age_of_vehicle"] = df["age_of_vehicle"].replace(-1, median).fillna(median)

  # Missing values are code as -1, changing to na

  df= df.replace([-1,"-1"], np.nan)


  return df

def pipelineTest1(dfCasualty, dfCollision, dfVehicle):

  test_list = []

  print("Test 1 Data Loading")

  # A Pass if the csv has the correct number of columns

  if dfCasualty.shape[1] == 23:
      print("PASS")
      test_list.append("PASS")
  else:
      print(f"FAIL: Expected 23 columns in Casualty, got {dfCasualty.shape[1]}")
      test_list.append("FAIL")

  if dfCollision.shape[1] == 44:
      print("PASS")
      test_list.append("PASS")
  else:
      print(f"FAIL: Expected 44 columns in Collision, got {dfCollision.shape[1]}")
      test_list.append("FAIL")

  if dfVehicle.shape[1] == 32:
      print("PASS")
      test_list.append("PASS")
  else:
      print(f"FAIL: Expected 32 columns in Vehicle, got {dfVehicle.shape[1]}")
      test_list.append("FAIL")

  print("Test 1 Complete")

  return test_list

def pipelineTest2(dfCasualtyProcessed, dfCollisionProcessed, dfVehicleProcessed, test_list = None):

  if test_list is None:
      test_list = []

  print("Test 2 Data Transforming")

  # A Fail if any of the age columns are negative

  if (dfCasualtyProcessed["age_of_casualty"]<0).any():
    print("FAIL: Negative age_of_casualty in Casualty")
    test_list.append("FAIL")
  else:
    print("PASS")
    test_list.append("PASS")

  if (dfVehicleProcessed["age_of_driver"]<0).any():
    print("FAIL: Negative age_of_driver in Vehicle")
    test_list.append("FAIL")
  else:
    print("PASS")
    test_list.append("PASS")

  if(dfVehicleProcessed["age_of_vehicle"]<0).any():
    print("FAIL: Negative age_of_vehicle in Vehicle")
    test_list.append("FAIL")
  else:
    print("PASS")
    test_list.append("PASS")

  print("Test 2 Complete")


  return test_list

def pipelineTest3(test_list = None):

  if test_list is None:
    test_list = []

  print("Test 3 Data Serving")

  # A Pass if each table has at least one row

  db_name = "uk_road_safety_data.db"

  # Making directory is where script is running

  try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
  except:
    script_dir = "."

  db_path = os.path.join(script_dir, db_name)

  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()


  tables = ["fact_vehicle", "dim_collision", "dim_vehicle", "dim_datetime"] # Each table name in db

  for table in tables:
    cursor.execute(f"SELECT 1 FROM {table} LIMIT 1;")  # Check if theres any data
    result = cursor.fetchone() # One row of Data
    if result is None:

      print(f"FAIL: {table} has no data")
      test_list.append("FAIL")

    else:
      print("PASS")
      test_list.append("PASS")

  print("Test 3 Complete")

  conn.close()

  return test_list

# Calling functions thats loads csv, describes dataset, creates visualisations and processes dataset for Casualty

object_columns = ["collision_index", "collision_ref_no", "lsoa_of_casualty"]
dfCasualty = loadTransform("casualty", object_columns) # Function for loading data

#fullDescribe(dfCasualty)  # Function describes dataset
#vizCasualty(dfCasualty)   # Function creates visualisations

dfCasualtyProcessed = processCasualty(dfCasualty) # Function processes data

# Calling functions thats loads csv, describes dataset, creates visualisations and processes dataset for Collision

object_columns = ["collision_index", "collision_ref_no", "date", "time",
                  "local_authority_ons_district", "local_authority_highway",
                  "local_authority_highway_current","lsoa_of_accident_location"]

dfCollision = loadTransform("collision",object_columns)

#fullDescribe(dfCollision)
#vizCollision(dfCollision)

dfCollisionProcessed = processCollision(dfCollision)

# Calling functions thats loads csv, describes dataset, creates visualisations and processes dataset for Vehicle

object_columns = ["collision_index","collision_ref_no","generic_make_model","lsoa_of_driver"]

dfVehicle = loadTransform("vehicle", object_columns)

#fullDescribe(dfVehicle)
#vizVehicle(dfVehicle)

dfVehicleProcessed = processVehicle(dfVehicle, most_popular_cars)

# Creating tables in sql using sqlite

db_name = "uk_road_safety_data.db"

# Making sure db file is created in script directory

try:
  script_dir = os.path.dirname(os.path.abspath(__file__))
except: # __file__ doesnt exist in colab
  script_dir = "."

db_path = os.path.join(script_dir, db_name)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()


# dim_casualty  - left in to show understanding of where I went wrong but is not created reflected in ERD

create_dim_casualty = '''
CREATE TABLE IF NOT EXISTS dim_casualty(

    collision_index TEXT,
    collision_year INT,
    collision_ref_no TEXT,
    vehicle_reference INT,
    casualty_reference INT,
    casualty_class INT,
    sex_of_casualty INT,
    is_male INT,
    age_of_casualty INT,
    age_band_of_casualty TEXT,
    casualty_severity INT,
    pedestrian_location INT,
    pedestrian_movement INT,
    car_passenger INT,
    bus_or_coach_passenger INT,
    pedestrian_road_maintenance_worker INT,
    casualty_type INT,
    casualty_imd_decile INT,
    lsoa_of_casualty TEXT,
    enhanced_casualty_severity INT,
    casualty_injury_based INT,
    casualty_adjusted_severity_serious FLOAT,
    casualty_adjusted_severity_slight FLOAT,
    casualty_distance_banding INT,
    PRIMARY KEY(collision_index, vehicle_reference, casualty_reference)

);

'''


# dim_collision

create_dim_collision = '''
CREATE TABLE IF NOT EXISTS dim_collision(

    collision_index TEXT PRIMARY KEY,
    collision_ref_no TEXT,
    location_easting_osgr FLOAT,
    location_northing_osgr FLOAT,
    longitude FLOAT,
    latitude FLOAT,
    police_force INT,
    collision_severity INT,
    number_of_vehicles INT,
    number_of_casualties INT,
    date TEXT,
    day_of_week INT,
    time TEXT,
    local_authority_ons_district TEXT,
    local_authority_highway TEXT,
    local_authority_highway_current TEXT,
    first_road_class INT,
    first_road_number INT,
    road_type INT,
    speed_limit INT,
    junction_detail INT,
    junction_control INT,
    second_road_class INT,
    second_road_number INT,
    pedestrian_crossing INT,
    light_conditions INT,
    weather_conditions INT,
    road_surface_conditions INT,
    special_conditions_at_site INT,
    carriageway_hazards INT,
    urban_or_rural_area INT,
    did_police_officer_attend_scene_of_accident INT,
    trunk_road_flag INT,
    lsoa_of_accident_location TEXT,
    collision_injury_based INT,
    collision_adjusted_severity_serious FLOAT,
    collision_adjusted_severity_slight FLOAT,
    casualty_per_vehicle FLOAT,
    normal_conditions INT

);

'''



# dim_vehicle

create_dim_vehicle = '''
CREATE TABLE IF NOT EXISTS dim_vehicle(

    collision_index TEXT,
    vehicle_reference INT,
    collision_year INT,
    collision_ref_no TEXT,
    vehicle_type INT,
    towing_and_articulation INT,
    vehicle_manoeuvre_historic INT,
    vehicle_manoeuvre INT,
    vehicle_direction_from INT,
    vehicle_direction_to INT,
    vehicle_location_restricted_lane_historic INT,
    vehicle_location_restricted_lane INT,
    junction_location INT,
    skidding_and_overturning INT,
    hit_object_in_carriageway INT,
    vehicle_leaving_carriageway INT,
    hit_object_off_carriageway INT,
    first_point_of_impact INT,
    vehicle_left_hand_drive INT,
    journey_purpose_of_driver_historic INT,
    journey_purpose_of_driver INT,
    sex_of_driver INT,
    age_of_driver INT,
    age_band_of_driver INT,
    engine_capacity_cc INT,
    propulsion_code INT,
    age_of_vehicle INT,
    generic_make_model TEXT,
    driver_imd_decile INT,
    lsoa_of_driver TEXT,
    escooter_flag INT,
    driver_distance_banding INT,
    brand TEXT,
    brand_code INT,
    PRIMARY KEY (collision_index, vehicle_reference));

 '''




# dim_datetime

# Extracting the date and time and day of week from the collision table

dim_datetime = pd.DataFrame()
dim_datetime['collision_date'] = dfCollisionProcessed['date']
dim_datetime['collision_time'] = dfCollisionProcessed['time']
dim_datetime['year'] = dfCollisionProcessed['date'].apply(lambda x: x.year)
dim_datetime['month'] = dfCollisionProcessed['date'].apply(lambda x: x.month)
dim_datetime['day'] = dfCollisionProcessed['date'].apply(lambda x: x.day)
dim_datetime['day_of_week'] = dfCollisionProcessed['day_of_week']
dim_datetime['hour'] = dfCollisionProcessed['time'].apply(lambda x: x.hour)
dim_datetime['minute'] = dfCollisionProcessed['time'].apply(lambda x: x.minute)


create_dim_datetime = '''
CREATE TABLE IF NOT EXISTS dim_datetime(

    datetime_id INTEGER PRIMARY KEY AUTOINCREMENT,
    collision_date TEXT,
    collision_time TEXT,
    year INT,
    month INT,
    day INT,
    hour INT,
    minute INT,
    day_of_week INT);

 '''




 # fact_vehicle

create_fact_vehicle = '''
CREATE TABLE IF NOT EXISTS fact_vehicle(
    collision_index TEXT,
    vehicle_reference INT,
    datetime_id INT,

    PRIMARY KEY (collision_index, vehicle_reference),
    FOREIGN KEY (collision_index) REFERENCES dim_collision(collision_index),
    FOREIGN KEY (collision_index, vehicle_reference) REFERENCES dim_vehicle(collision_index, vehicle_reference),
    FOREIGN KEY (datetime_id) REFERENCES dim_datetime(datetime_id)
);

'''

# Executing and committing

cursor.execute("PRAGMA foreign_keys = ON;")
#cursor.execute(create_dim_casualty) # Commented out to reflect change in ERD
cursor.execute(create_dim_collision)
cursor.execute(create_dim_vehicle)
cursor.execute(create_dim_datetime)
cursor.execute(create_fact_vehicle)
conn.commit()

# Loading data into database

# dfCasualtyProcessed.to_sql('dim_casualty', conn, if_exists='append', index=False) # Commented out to reflect change in ERD
dfCollisionProcessed.to_sql('dim_collision', conn, if_exists='append', index=False)
dfVehicleProcessed.to_sql('dim_vehicle', conn, if_exists='append', index=False)
dim_datetime.to_sql('dim_datetime', conn, if_exists='append', index=False)

# Joining datasets to create fact_vehicle dataframe

"""
# Previous implementation of Casualty

fact_vehicle = dfVehicleProcessed.merge(
    dfCasualtyProcessed,
    on=['collision_index', 'vehicle_reference'],
    how='inner'
)

"""

# Merging Collision

fact_vehicle = dfVehicleProcessed.merge(
    dfCollisionProcessed[['collision_index', 'date', 'time']],
    on='collision_index',
    how='left'
)

# Merging datetime

dim_datetime_db = pd.read_sql_query("SELECT * FROM dim_datetime", conn) # Need to get autoincremented datetime_id


# Ensure type consistency for merging date/time

fact_vehicle['date_str'] = pd.to_datetime(fact_vehicle['date']).dt.date.astype(str)
fact_vehicle['time_str'] = pd.to_datetime(fact_vehicle['time'], format='%H:%M:%S', errors='coerce').dt.time.astype(str)

dim_datetime_db['collision_date_str'] = pd.to_datetime(dim_datetime_db['collision_date']).dt.date.astype(str)
dim_datetime_db['collision_time_str'] = pd.to_datetime(dim_datetime_db['collision_time']).dt.time.astype(str)

# Merging to get datetime_id (changed to inner to ensure datetime_id is not null)
fact_vehicle = fact_vehicle.merge(
    dim_datetime_db[['datetime_id', 'collision_date_str', 'collision_time_str']],
    left_on=['date_str', 'time_str'],
    right_on=['collision_date_str', 'collision_time_str'],
    how='inner'
)

# Keeping columns needed for fact table

fact_vehicle = fact_vehicle[['collision_index', 'vehicle_reference', 'datetime_id']]
fact_vehicle = fact_vehicle.drop_duplicates(subset=['collision_index', 'vehicle_reference']) # Duplicates exist from merging
fact_vehicle.to_sql('fact_vehicle', conn, if_exists='append', index=False)


conn.close() # Close connection after all operations are done

# Pipeline Testing

# Each test function prints result of each test but also returns a list

test_list = pipelineTest1(dfCasualty,dfCollision,dfVehicle) # Function for testing csv loading
test_list = pipelineTest2(dfCasualtyProcessed,dfCollisionProcessed,dfVehicleProcessed, test_list) # Function for testing data transforming
test_list = pipelineTest3(test_list) # Function for testing db file
print("\nFull Test Results: ")
print(test_list)


if "FAIL" in test_list:
    print("One or more tests failed")
else:
    print("All Pipeline Tests Passed")

