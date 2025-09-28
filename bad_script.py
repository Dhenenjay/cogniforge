
import sys

print("Executing bad script...")
print("This will fail...")
raise Exception("Simulated failure!")
sys.exit(1)
