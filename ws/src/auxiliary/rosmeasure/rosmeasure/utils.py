#!/usr/bin/env python
# utils.py
"""Utilities for measuring delay within nodes.
"""
######################
# Imports & Globals
######################

# ROS Python package
import rospy

# Timing
import time


# Parameters
DISABLED = False


# On start check for /delay_measure
try:
    TIMING = str(rospy.get_param("/delay_measure", False)).lower() == "true"

    #rospy.loginfo("Delay measurement %s" % ("[Enabled]" if TIMING else "[Disabled]"))

    if not TIMING:
        DISABLED = True
except:
    pass


######################
# conddisable decorator
######################

def conddisable():
    """Decorator which disables functions when DISABLED."""
    global DISABLED

    def block(f):
        return lambda *x, **y: None
 
    def let_pass(f):
        return f

    return block if DISABLED else let_pass


######################
# Measurer class
######################

class Measurer:
    def __init__(self, name = "", unit = ""):
        self._name = name if name != "" else "Measurer"
        self._unit = unit if unit != "" else "s"
        self._start = 0
        self._count = 0
        self._sum = 0
        self._max = 0
        self._last = 0


    def unitExp(self, unit):
        """Gets power of unit."""
        if unit == "s":
            return 0
        elif unit == "ms":
            return -3
        elif unit == "us":
            return -6
        elif unit == "ns":
            return -9
        else:
            return 0


    def convertUnits(self, value, unit_from, unit_to):
        """Converts unit of measured value."""
        f = self.unitExp(unit_from)
        t = self.unitExp(unit_to)

        if f == t:
            return value
        else:
            return value * pow(10, f - t)


    @conddisable()
    def updateStatistics(self):
        self._count += 1
        self._sum += self._last
        self._max = max(self._max, self._last)


    @conddisable()
    def summary(self):
        print "%s: cur=%.4f%s avg=%.4f%s max=%.4f%s" % (
            self._name, self._last, self._unit,
            self._sum / self._count, self._unit,
            self._max, self._unit
        )


######################
# TimeMeasurer class
######################

class TimeMeasurer(Measurer):
    """Class for TimeMeasurer, that records time spend in selected section."""

    def __init__(self, *args, **kwargs):
        Measurer.__init__(self, *args, **kwargs)


    @conddisable()
    def start(self):
        """Start a measurement."""
        self._start = time.time()


    @conddisable()
    def end(self):
        """End a measurement."""
        self._last = self.convertUnits(
            time.time() - self._start,
            "s",
            self._unit
        )

        self.updateStatistics()


######################
# DelayMeasurer class
######################

class DelayMeasurer(Measurer):
    """Class for DelayMeasurer, that records delays between
    timestamps and current time."""

    def __init__(self, *args, **kwargs):
        Measurer.__init__(self, *args, **kwargs)


    @conddisable()
    def delay(self, header):
        """Records a delay between current time and header."""
        self._last = self.convertUnits(
            (rospy.Time.now() - header.stamp).to_sec(),
            "s",
            self._unit
        )

        self.updateStatistics()

