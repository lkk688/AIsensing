import adi
import sys

print("Resetting TDD state...")
try:
    tdd = adi.tddn("ip:192.168.2.1")
    tdd.enable = False
    print("TDD Disabled successfully.")
except Exception as e:
    print(f"Error resetting TDD: {e}")
