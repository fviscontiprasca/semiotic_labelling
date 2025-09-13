# Load the actual classes from text file
with open('data/oid_urban/actual_classes.txt', 'r') as f:
    available_classes = [line.strip() for line in f.readlines()]

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
found_current = [cls for cls in current_classes if cls in available_classes]
print("Current classes found in actual_classes.txt:")
for item in found_current:
    print(f"  ✓ {item}")

missing_current = [cls for cls in current_classes if cls not in available_classes]
if missing_current:
    print(f"\nCurrent classes NOT found in actual_classes.txt:")
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

found_additional = [cls for cls in additional_terms if cls in available_classes]
print(f"\n\nAdditional urban/architectural classes found in actual_classes.txt:")
for item in sorted(found_additional):
    print(f"  + {item}")

print(f"\nTotal additional found: {len(found_additional)}")

# Combine all found classes
all_found = set(found_current + found_additional)
print(f"\nTotal combined urban/architectural classes: {len(all_found)}")

# Save combined list to a file
with open('my_classes.txt', 'w') as f:
    for cls in sorted(all_found):
        f.write(f"{cls}\n")

print(f"\nCombined list saved to 'my_classes.txt'")