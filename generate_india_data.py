"""Generate comprehensive India-wide accident heatmap data covering all states."""
import json, random, math

random.seed(42)

# Major Indian cities/regions with (lat, lng, weight, radius)
# weight = relative accident density, radius = spread area
locations = [
    # SOUTH INDIA (higher density per request)
    (13.08, 80.27, 120, 0.15),  # Chennai
    (12.97, 77.59, 130, 0.18),  # Bangalore
    (17.39, 78.49, 110, 0.16),  # Hyderabad
    (10.85, 76.27, 60, 0.12),   # Palakkad/Kerala central
    (9.93, 76.26, 80, 0.14),    # Kochi
    (8.52, 76.94, 70, 0.12),    # Trivandrum
    (11.02, 76.97, 65, 0.12),   # Coimbatore
    (9.92, 78.12, 55, 0.10),    # Madurai
    (11.94, 79.81, 45, 0.08),   # Pondicherry
    (10.79, 78.70, 50, 0.10),   # Trichy
    (13.63, 79.42, 40, 0.08),   # Tirupati
    (15.35, 75.12, 45, 0.10),   # Hubli
    (12.30, 76.66, 40, 0.08),   # Mysore
    (15.83, 78.04, 35, 0.08),   # Kurnool
    (16.51, 80.65, 60, 0.12),   # Vijayawada
    (17.73, 83.30, 55, 0.10),   # Visakhapatnam
    (14.68, 77.60, 35, 0.08),   # Anantapur
    (11.66, 78.16, 40, 0.08),   # Salem
    (8.73, 77.69, 35, 0.08),    # Tirunelveli
    (12.52, 74.99, 40, 0.08),   # Mangalore
    (15.44, 73.88, 45, 0.08),   # Goa
    (10.52, 76.21, 50, 0.10),   # Thrissur
    (11.25, 75.77, 55, 0.10),   # Kozhikode
    (12.50, 75.00, 30, 0.06),   # Kasaragod
    
    # NORTH INDIA
    (28.61, 77.23, 150, 0.20),  # Delhi NCR
    (26.85, 80.95, 90, 0.15),   # Lucknow
    (26.45, 80.35, 60, 0.10),   # Kanpur
    (25.32, 82.99, 50, 0.10),   # Varanasi
    (27.18, 78.02, 45, 0.08),   # Agra
    (26.92, 75.79, 80, 0.14),   # Jaipur
    (30.73, 76.78, 65, 0.12),   # Chandigarh
    (31.63, 74.87, 55, 0.10),   # Amritsar
    (30.90, 75.86, 60, 0.10),   # Ludhiana
    (28.98, 77.71, 40, 0.08),   # Meerut
    (29.97, 76.88, 35, 0.06),   # Karnal
    (28.40, 77.31, 45, 0.08),   # Faridabad
    (28.47, 77.03, 50, 0.08),   # Gurugram
    (25.43, 81.85, 40, 0.08),   # Allahabad
    (26.84, 75.76, 35, 0.06),   # Ajmer
    (24.58, 73.69, 30, 0.06),   # Udaipur
    (26.29, 73.02, 35, 0.06),   # Jodhpur
    
    # WEST INDIA
    (19.08, 72.88, 140, 0.20),  # Mumbai
    (18.52, 73.86, 100, 0.16),  # Pune
    (21.17, 72.83, 60, 0.12),   # Surat
    (23.02, 72.57, 80, 0.14),   # Ahmedabad
    (22.31, 73.18, 50, 0.10),   # Vadodara
    (21.76, 72.15, 35, 0.06),   # Bhavnagar
    (22.72, 75.86, 55, 0.10),   # Indore
    (23.26, 77.41, 50, 0.10),   # Bhopal
    (21.15, 79.09, 55, 0.10),   # Nagpur
    (19.88, 75.34, 40, 0.08),   # Aurangabad
    (19.99, 73.78, 45, 0.08),   # Nashik
    (16.70, 74.24, 35, 0.06),   # Kolhapur
    
    # EAST INDIA
    (22.57, 88.36, 110, 0.16),  # Kolkata
    (25.61, 85.14, 65, 0.12),   # Patna
    (20.30, 85.83, 50, 0.10),   # Bhubaneswar
    (23.35, 85.33, 45, 0.10),   # Ranchi
    (22.80, 86.20, 40, 0.08),   # Jamshedpur
    (25.59, 85.10, 35, 0.06),   # Darbhanga
    (21.49, 86.94, 30, 0.06),   # Balasore
    
    # NORTHEAST
    (26.14, 91.74, 40, 0.08),   # Guwahati
    (25.58, 91.88, 25, 0.06),   # Shillong
    (23.83, 91.28, 25, 0.06),   # Agartala
    (25.67, 94.11, 20, 0.06),   # Imphal
    (27.33, 88.61, 20, 0.05),   # Gangtok
    (26.76, 94.22, 20, 0.05),   # Dibrugarh
    
    # CENTRAL
    (23.18, 79.95, 40, 0.08),   # Jabalpur
    (21.25, 81.63, 35, 0.08),   # Raipur
    (24.80, 79.93, 25, 0.06),   # Sagar
    
    # Highways / corridors (linear scatter)
    (19.50, 75.00, 30, 0.20),   # NH48 corridor
    (15.00, 76.50, 25, 0.15),   # NH44 South segment
    (22.00, 77.00, 25, 0.15),   # Central MP
    (27.00, 79.50, 25, 0.12),   # UP corridor
    (24.00, 85.00, 20, 0.12),   # Bihar-JH corridor
    (16.00, 80.00, 25, 0.15),   # AP coastal
    (10.00, 77.50, 30, 0.12),   # TN interior
    (13.50, 77.00, 20, 0.10),   # KA interior
]

points = []
for lat, lng, weight, radius in locations:
    n = int(weight * 1.5)
    for _ in range(n):
        # Gaussian spread
        dlat = random.gauss(0, radius * 0.5)
        dlng = random.gauss(0, radius * 0.5)
        severity = random.choices([1, 2, 3], weights=[0.15, 0.50, 0.35])[0]
        intensity = {1: 1.0, 2: 0.6, 3: 0.3}[severity]
        points.append({
            "lat": round(lat + dlat, 5),
            "lng": round(lng + dlng, 5),
            "severity": severity,
            "intensity": intensity
        })

# Add scattered highway points across India
highway_routes = [
    # NH44 (North-South) Delhi to Kanyakumari
    [(28.6, 77.2), (27.5, 77.5), (26.8, 75.8), (25.3, 78.0), (23.2, 77.4),
     (21.1, 79.1), (18.5, 73.9), (17.4, 78.5), (15.4, 75.1), (13.1, 80.3),
     (11.0, 77.0), (9.9, 78.1), (8.5, 77.0)],
    # NH48 Mumbai to Delhi
    [(19.1, 72.9), (20.0, 73.0), (21.2, 72.8), (22.3, 73.2), (23.0, 72.6),
     (24.5, 74.0), (26.0, 74.5), (27.0, 75.5), (28.4, 77.0)],
    # East Coast
    [(13.1, 80.3), (14.5, 80.0), (16.5, 80.7), (17.7, 83.3), (19.3, 84.8),
     (20.3, 85.8), (21.5, 87.0), (22.6, 88.4)],
    # NH27 East-West
    [(26.1, 91.7), (25.6, 85.1), (26.5, 80.4), (26.8, 80.9), (26.9, 75.8),
     (23.0, 72.6)],
]

for route in highway_routes:
    for i in range(len(route)-1):
        lat1, lng1 = route[i]
        lat2, lng2 = route[i+1]
        for t in range(12):
            frac = t / 12.0
            lat = lat1 + (lat2-lat1) * frac + random.gauss(0, 0.08)
            lng = lng1 + (lng2-lng1) * frac + random.gauss(0, 0.08)
            severity = random.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]
            points.append({
                "lat": round(lat, 5),
                "lng": round(lng, 5),
                "severity": severity,
                "intensity": {1: 1.0, 2: 0.6, 3: 0.3}[severity]
            })

print(f"Generated {len(points)} data points across India")

# State-level statistics
state_stats = [
    {"state": "Tamil Nadu", "code": "TN", "accidents": 57228, "fatalities": 16685, "injuries": 62290, "risk": "High"},
    {"state": "Madhya Pradesh", "code": "MP", "accidents": 49587, "fatalities": 11073, "injuries": 51397, "risk": "High"},
    {"state": "Uttar Pradesh", "code": "UP", "accidents": 38297, "fatalities": 22256, "injuries": 33586, "risk": "Critical"},
    {"state": "Karnataka", "code": "KA", "accidents": 37852, "fatalities": 10856, "injuries": 44297, "risk": "High"},
    {"state": "Maharashtra", "code": "MH", "accidents": 29216, "fatalities": 12263, "injuries": 25780, "risk": "High"},
    {"state": "Kerala", "code": "KL", "accidents": 39420, "fatalities": 4317, "injuries": 42770, "risk": "Medium"},
    {"state": "Rajasthan", "code": "RJ", "accidents": 24239, "fatalities": 10979, "injuries": 23646, "risk": "High"},
    {"state": "Andhra Pradesh", "code": "AP", "accidents": 21975, "fatalities": 8292, "injuries": 24378, "risk": "High"},
    {"state": "Telangana", "code": "TS", "accidents": 19797, "fatalities": 7150, "injuries": 19923, "risk": "High"},
    {"state": "Gujarat", "code": "GJ", "accidents": 17804, "fatalities": 7302, "injuries": 13948, "risk": "Medium"},
    {"state": "West Bengal", "code": "WB", "accidents": 12687, "fatalities": 5363, "injuries": 10754, "risk": "Medium"},
    {"state": "Bihar", "code": "BR", "accidents": 9068, "fatalities": 7092, "injuries": 7672, "risk": "High"},
    {"state": "Haryana", "code": "HR", "accidents": 9798, "fatalities": 5765, "injuries": 10432, "risk": "Medium"},
    {"state": "Punjab", "code": "PB", "accidents": 5877, "fatalities": 4347, "injuries": 7233, "risk": "Medium"},
    {"state": "Odisha", "code": "OD", "accidents": 10384, "fatalities": 5333, "injuries": 8971, "risk": "Medium"},
    {"state": "Jharkhand", "code": "JH", "accidents": 4852, "fatalities": 3728, "injuries": 3766, "risk": "Medium"},
    {"state": "Assam", "code": "AS", "accidents": 6802, "fatalities": 2641, "injuries": 6724, "risk": "Medium"},
    {"state": "Chhattisgarh", "code": "CG", "accidents": 11562, "fatalities": 3846, "injuries": 11088, "risk": "Medium"},
    {"state": "Delhi", "code": "DL", "accidents": 5610, "fatalities": 1358, "injuries": 5639, "risk": "Medium"},
    {"state": "Goa", "code": "GA", "accidents": 3174, "fatalities": 334, "injuries": 3024, "risk": "Low"},
]

# News/incident data
recent_incidents = [
    {"date": "2025-12-15", "location": "Chennai-Bangalore Highway, TN", "lat": 12.82, "lng": 78.70, "type": "Multi-vehicle pileup", "severity": "Fatal", "casualties": 5, "cause": "Fog"},
    {"date": "2025-12-10", "location": "Outer Ring Road, Hyderabad", "lat": 17.44, "lng": 78.38, "type": "Truck collision", "severity": "Severe", "casualties": 3, "cause": "Speeding"},
    {"date": "2025-12-08", "location": "NH44 near Kurnool, AP", "lat": 15.83, "lng": 78.00, "type": "Bus accident", "severity": "Fatal", "casualties": 8, "cause": "Tire burst"},
    {"date": "2025-12-05", "location": "Kozhikode, Kerala", "lat": 11.25, "lng": 75.78, "type": "Two-wheeler crash", "severity": "Moderate", "casualties": 2, "cause": "Rain"},
    {"date": "2025-11-28", "location": "Mumbai-Pune Expressway", "lat": 18.85, "lng": 73.40, "type": "Chain collision", "severity": "Fatal", "casualties": 4, "cause": "Fog"},
    {"date": "2025-11-22", "location": "Yamuna Expressway, UP", "lat": 27.80, "lng": 77.60, "type": "Car overturn", "severity": "Severe", "casualties": 3, "cause": "Speeding"},
    {"date": "2025-11-15", "location": "NH75 near Mysore, KA", "lat": 12.30, "lng": 76.65, "type": "Motorcycle accident", "severity": "Moderate", "casualties": 1, "cause": "Pothole"},
    {"date": "2025-11-10", "location": "ECR, Chennai", "lat": 12.85, "lng": 80.23, "type": "Pedestrian hit", "severity": "Fatal", "casualties": 1, "cause": "Drunk driving"},
    {"date": "2025-11-05", "location": "Jaipur-Delhi Highway", "lat": 27.50, "lng": 76.50, "type": "Truck rollover", "severity": "Severe", "casualties": 2, "cause": "Overloading"},
    {"date": "2025-10-30", "location": "Coimbatore-Salem NH", "lat": 11.35, "lng": 77.40, "type": "Bus-truck collision", "severity": "Fatal", "casualties": 6, "cause": "Wrong side driving"},
    {"date": "2025-10-25", "location": "Kolkata EM Bypass", "lat": 22.52, "lng": 88.40, "type": "Hit and run", "severity": "Moderate", "casualties": 1, "cause": "Negligence"},
    {"date": "2025-10-20", "location": "Bangalore Hebbal Flyover", "lat": 13.04, "lng": 77.59, "type": "Car collision", "severity": "Severe", "casualties": 2, "cause": "Signal violation"},
]

with open("backend/heatmap_data.json", "w") as f:
    json.dump(points, f)
with open("backend/state_stats.json", "w") as f:
    json.dump(state_stats, f)
with open("backend/recent_incidents.json", "w") as f:
    json.dump(recent_incidents, f)

print(f"Saved {len(points)} heatmap points")
print(f"Saved {len(state_stats)} state statistics")
print(f"Saved {len(recent_incidents)} recent incidents")
