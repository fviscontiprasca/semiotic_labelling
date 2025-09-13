import pandas as pd

# Load the OIDv7 class descriptions
df = pd.read_csv('data/oid_urban/oidv7-class-descriptions.csv')

# Current classes in my_classes.txt
current_classes = [
    'Building', 'House', 'Skyscraper', 'Tower', 'Castle', 'Office building', 
    'Convenience store', 'Lighthouse', 'Tree house', 'Window', 'Door', 
    'Door handle', 'Window blind', 'Stairs', 'Porch', 'Street light', 
    'Traffic light', 'Traffic sign', 'Stop sign', 'Bench', 'Fountain', 
    'Sculpture', 'Bronze sculpture', 'Lamp', 'Vehicle', 'Car', 'Bus', 
    'Truck', 'Motorcycle', 'Bicycle', 'Golf cart', 'Tree', 'Plant', 
    'Houseplant', 'Flower', 'Flowerpot', 'Palm tree', 'Christmas tree', 
    'Common sunflower', 'Person', 'Parking meter', 'Cart', 'Tent', 'Fire hydrant'
]

# Check which current classes are found
found_current = df[df['DisplayName'].isin(current_classes)]
print("Current classes found in OIDv7:")
for item in found_current['DisplayName'].tolist():
    print(f"  ✓ {item}")

missing_current = [cls for cls in current_classes if cls not in found_current['DisplayName'].tolist()]
if missing_current:
    print(f"\nCurrent classes NOT found in OIDv7:")
    for item in missing_current:
        print(f"  ✗ {item}")

print(f"\nTotal current found: {len(found_current)}/{len(current_classes)}")

# Search for additional architectural and urban classes
additional_terms = [
    'Bridge', 'Stadium', 'Park', 'Street', 'Road', 'Sidewalk', 'Plaza', 
    'Square', 'Church', 'Museum', 'Library', 'School', 'Hospital', 'Airport', 
    'Station', 'Mall', 'Market', 'Restaurant', 'Cafe', 'Hotel', 'Apartment', 
    'Balcony', 'Roof', 'Garage', 'Fence', 'Gate', 'Wall', 'Column', 'Pillar', 
    'Dome', 'Spire', 'Arch', 'Construction', 'Crane', 'Scaffolding', 'Monument', 
    'Statue', 'Obelisk', 'Pyramid', 'Temple', 'Mosque', 'Synagogue', 'Cathedral', 
    'Chapel', 'Palace', 'Mansion', 'Villa', 'Cottage', 'Cabin', 'Warehouse', 
    'Factory', 'Silo', 'Chimney', 'Antenna', 'Billboard', 'Pedestrian', 
    'Crosswalk', 'Intersection', 'Neighbourhood', 'Downtown', 'Suburb', 
    'Metro', 'Train station', 'Bus station', 'Parking lot', 'Gas station'
]

found_additional = df[df['DisplayName'].isin(additional_terms)]
print(f"\n\nAdditional urban/architectural classes found in OIDv7:")
for item in sorted(found_additional['DisplayName'].tolist()):
    print(f"  + {item}")

print(f"\nTotal additional found: {len(found_additional)}")

# Combine all found classes
all_found = set(found_current['DisplayName'].tolist() + found_additional['DisplayName'].tolist())
print(f"\nTotal combined urban/architectural classes: {len(all_found)}")

# Save combined list to a file
with open('urban_classes_oidv7.txt', 'w') as f:
    for cls in sorted(all_found):
        f.write(f"{cls}\n")

print(f"\nCombined list saved to 'urban_classes_oidv7.txt'")