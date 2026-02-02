from astral.sun import sun
from datetime import datetime, time
import re

# use in 1_hdf_to_ppi.py
def is_daytime_file(file_name, radar_location):
    # File name should have date time: 0000ISR-PPIVol-20140427-144102-0a0a
    match = re.search(r'(\d{8})-(\d{6})', file_name)
    if match:
        time_str = match.group(2)
        print(time_str)
        hour, minute, second = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
        file_time = time(hour, minute, second)

        # Extract date from the file name
        date_str = match.group(1)
        print(date_str)
        file_date = datetime.strptime(date_str, "%Y%m%d").date()

        # Calculate sunrise and sunset for that date
        sun_times = sun(radar_location.observer, date=file_date)
        sunrise = sun_times['sunrise'].time()
        sunset = sun_times['sunset'].time()

        # Check if the file time is within daytime
        return sunrise <= file_time <= sunset
    return False
