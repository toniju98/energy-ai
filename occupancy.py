class OccupancySimulator:
    """
    Simulates occupancy patterns for a home.
    Uses a simple model based on time of day and day of week.
    """
    def __init__(self):
        # Define typical occupancy patterns (0-1 scale, where 1 means fully occupied)
        # Weekday patterns (Monday-Friday)
        self.weekday_pattern = {
            0: 0.1,   # 12 AM - 6 AM: Mostly sleeping
            1: 0.1,
            2: 0.1,
            3: 0.1,
            4: 0.1,
            5: 0.1,
            6: 0.3,   # 6 AM - 9 AM: Morning routine
            7: 0.3,
            8: 0.3,
            9: 0.1,   # 9 AM - 5 PM: Work hours
            10: 0.1,
            11: 0.1,
            12: 0.1,
            13: 0.1,
            14: 0.1,
            15: 0.1,
            16: 0.1,
            17: 0.3,  # 5 PM - 10 PM: Evening activities
            18: 0.8,
            19: 0.8,
            20: 0.8,
            21: 0.8,
            22: 0.5,  # 10 PM - 12 AM: Getting ready for bed
            23: 0.3
        }
        
        # Weekend patterns (Saturday-Sunday)
        self.weekend_pattern = {
            0: 0.1,   # 12 AM - 8 AM: Sleeping
            1: 0.1,
            2: 0.1,
            3: 0.1,
            4: 0.1,
            5: 0.1,
            6: 0.1,
            7: 0.1,
            8: 0.3,   # 8 AM - 10 PM: Various activities
            9: 0.5,
            10: 0.7,
            11: 0.8,
            12: 0.8,
            13: 0.8,
            14: 0.8,
            15: 0.8,
            16: 0.8,
            17: 0.8,
            18: 0.8,
            19: 0.8,
            20: 0.8,
            21: 0.8,
            22: 0.5,  # 10 PM - 12 AM: Getting ready for bed
            23: 0.3
        }
    
    def get_occupancy(self, hour, day):
        """
        Get occupancy level for a given hour and day.
        
        Args:
            hour (int): Hour of the day (0-23)
            day (int): Day of the week (0-6, where 0 is Monday)
            
        Returns:
            float: Occupancy level (0-1)
        """
        # Use weekend pattern for Saturday (5) and Sunday (6)
        if day >= 5:  # Weekend
            return self.weekend_pattern[hour]
        else:  # Weekday
            return self.weekday_pattern[hour] 