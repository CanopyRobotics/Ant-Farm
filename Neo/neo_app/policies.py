# Thin adapters to reuse Ant-Farm policies without duplication
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from slotting import AffinitySlotting, PopularityABCSlotting, RandomSlotting, RoundRobinSlotting
