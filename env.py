import numpy as np
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from occupancy import OccupancySimulator

class WeatherAPI:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("Please set OPENWEATHER_API_KEY in your .env file")
        
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.cache = {}  # Cache weather data to avoid too many API calls
        self.cache_duration = timedelta(minutes=30)  # Cache weather data for 30 minutes
    
    def get_temperature(self, hour, day):
        """
        Get real temperature data from OpenWeatherMap API 2.5.
        Uses caching to avoid excessive API calls.
        
        Args:
            hour (int): Current hour (0-23)
            day (int): Current day (0-6)
            
        Returns:
            float: Current temperature in Celsius
        """
        current_time = datetime.now()
        cache_key = f"{current_time.date()}_{hour}"
        
        # Check if we have cached data that's still valid
        if cache_key in self.cache:
            cache_time, temp = self.cache[cache_key]
            if current_time - cache_time < self.cache_duration:
                return temp
        
        # If no valid cache, fetch new data
        try:
            # Get coordinates for your location (you can modify these)
            lat = os.getenv('LATITUDE', '40.7128')  # Default to New York
            lon = os.getenv('LONGITUDE', '-74.0060')
            
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'  # Use Celsius
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            temperature = data['main']['temp']
            
            # Cache the result
            self.cache[cache_key] = (current_time, temperature)
            
            return temperature
            
        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            # Fallback to a reasonable temperature if API fails
            return 20.0

class ThermostatEnv:
    def __init__(self):
        self.temperature = 22.0  # Initial temperature
        self.day = 0  # 0-6 (Mon-Sun)
        self.hour = 0  # 0-23
        self.energy_used = 0
        self.weather = WeatherAPI()
        self.occupancy = OccupancySimulator()
        self.heat_transfer_coefficient = 0.1  # How quickly external temperature affects indoor
        
        # Comfort temperature ranges based on occupancy
        self.comfort_range = {
            'occupied': (20, 22),    # Comfortable range when occupied
            'unoccupied': (18, 24)   # Wider range when unoccupied
        }
    
    def reset(self):
        """Reset the environment to initial state"""
        self.temperature = 22.0
        self.day = 0
        self.hour = 0
        self.energy_used = 0
        return (self.temperature, self.day, self.hour,
                self.weather.get_temperature(self.hour, self.day),
                self.occupancy.get_occupancy(self.hour, self.day))

    def step(self, action):
        # action: 0 = do nothing, 1 = heating, 2 = cooling
        reward = 0
        energy_cost = 0

        # Get external temperature and occupancy
        external_temp = self.weather.get_temperature(self.hour, self.day)
        occupancy_level = self.occupancy.get_occupancy(self.hour, self.day)

        # Natural temperature change due to external temperature
        temp_diff = external_temp - self.temperature
        self.temperature += temp_diff * self.heat_transfer_coefficient

        # Action effects
        if action == 1:
            self.temperature = min(30, self.temperature + 1)  # Cap at 30°C
            energy_cost = 1
            self.energy_used += 1
        elif action == 2:
            self.temperature = max(15, self.temperature - 1)  # Floor at 15°C
            energy_cost = 1
            self.energy_used += 1

        # Comfort reward based on occupancy
        if occupancy_level > 0.5:  # High occupancy
            min_temp, max_temp = self.comfort_range['occupied']
        else:  # Low occupancy
            min_temp, max_temp = self.comfort_range['unoccupied']
        
        if min_temp <= self.temperature <= max_temp:
            comfort_reward = 1 * occupancy_level  # Scale reward by occupancy
        else:
            comfort_reward = -1 * occupancy_level  # Scale penalty by occupancy
            
        # Energy efficiency reward
        # Penalize energy usage more when temperature is already in comfort zone
        if min_temp <= self.temperature <= max_temp:
            energy_penalty = -0.5 * energy_cost  # Higher penalty for using energy when comfortable
        else:
            energy_penalty = -0.2 * energy_cost  # Lower penalty when trying to reach comfort zone
            
        # Total reward combines comfort and energy efficiency
        reward = comfort_reward + energy_penalty

        self.hour += 1
        if self.hour == 24:
            self.hour = 0
            self.day = (self.day + 1) % 7
        
        # Return state including external temperature and occupancy
        return (self.temperature, self.day, self.hour, external_temp, occupancy_level), reward
