import pandas as pd
import numpy as np
from datetime import datetime


class Parser:
    def __init__(self, filename_confirmed,
                 filename_deaths,
                 filename_recovered,
                 filename_population):
        self.confirmed = self.read_csv(filename_confirmed)
        self.deaths = self.read_csv(filename_deaths)
        self.recovered = self.read_csv(filename_recovered)
        self.population = self.read_population(filename_population)
        self.countries = list(np.intersect1d(self.confirmed.columns.values,
                                             self.population.index.values))

    def read_csv(self, filename):
        # Create pandas dataframe from .csv
        data = pd.read_csv(filename)

        # Manipulate the dataframe to have dates as row indices and country
        # names as column names
        data = data.set_index("Country/Region")
        data = data.T
        data.index = pd.to_datetime(data.index)

        return data

    def parse_data(self, start_date, end_date, country):
        self.validate_date(start_date)
        self.validate_date(end_date)
        self.validate_country(country)

        delta_i = self.confirmed.loc[:end_date, country].diff().dropna()
        delta_i = delta_i.astype(int)
        r = (self.deaths.loc[:end_date, country]
             + self.recovered.loc[:end_date, country])
        delta_r = r.diff().dropna().astype(int)
        i = (delta_i - delta_r).cumsum()
        return i[start_date:], r[start_date:]

    def read_population(self, filename):
        # Create pandas dataframe from .csv
        data = pd.read_csv(filename)
        data = data.set_index("Country")

        return data

    def parse_population(self, country):
        population = self.population.loc[country, "Population"]

        return population

    def validate_date(self, date_text):
        try:
            datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD!")

    def validate_country(self, country):
        if country not in self.countries:
            raise ValueError("Country not in list!")
