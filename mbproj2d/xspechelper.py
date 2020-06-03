# Copyright (C) 2016 Jeremy Sanders <jeremy@jeremysanders.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Module to interrogate xspec to get count rates and luminosities
given model parameters.

"""

import subprocess
import os
import select
import atexit
import re
import sys
import signal

def deleteFile(filename):
    """Delete file, ignoring errors."""
    try:
        os.unlink(filename)
    except OSError:
        pass

# keep track of xspec invocations which need finishing
_finishatexit = []

# tcl code to do an infinite evaluation of commands until end
tclloop = '''
autosave off
while { 1 } {
 set s [gets stdin]
 if { [eof stdin] } {
   tclexit
 }
 eval $s
}
'''

class XSpecHelper:
    """A helper to get count rates for temperature and densities."""

    specialcode = '@S@T@V'
    specialre = re.compile('%s (.*) %s' % (specialcode, specialcode))

    def __init__(self):
        try:
            self.xspecsub = subprocess.Popen(
                ['xspec'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )
        except OSError:
            raise RuntimeError('Failed to start xspec')

        self.throwAwayOutput()
        self.tempoutput = None
        _finishatexit.append(self)

        self.write('set SCODE %s\n' % self.specialcode)

        # debugging
        logfile = os.path.join(os.environ['HOME'], 'xspec.log.%i' % id(self))
        deleteFile(logfile)
        #self.xspecsub.stdin.write('log %s\n' % logfile)

        # form of xspec loop where it doesn't blow up if this program
        # closes its stdin
        self.write(tclloop)

    def write(self, text):
        self.xspecsub.stdin.write(text)#.encode('utf-8'))

    def throwAwayOutput(self):
        """Ignore output from program until no more data available."""
        while True:
            i, o, e = select.select([self.xspecsub.stdout.fileno()], [], [], 0)
            if i:
                t = os.read(i[0], 1024)
                if not t:  # file was closed
                    break
            else:
                break

    def readResult(self):
        """Return result from xspec."""
        search = None
        while not search:
            line = self.xspecsub.stdout.readline()
            #line = line.decode('utf-8')
            search = XSpecHelper.specialre.search(line)
        return search.group(1)

    def setAbund(self, abund):
        """Set model abundance."""
        self.write('abund %s\n' % abund)
        self.throwAwayOutput()

    def getRate(self):
        """Get rate from current model in current noticed band."""
        self.write('puts "$SCODE [tcloutr rate 1] $SCODE"\n')
        retn = self.readResult()
        modelrate = float( retn.split()[2] )
        return modelrate

    def getFlux(self, emin_keV, emax_keV):
        """Get flux from current model in energy range given."""
        self.write('flux %e %e\n' % (emin_keV, emax_keV))
        self.write('puts "$SCODE [tcloutr flux] $SCODE"\n')
        flux = float( self.readResult().split()[0] )
        return flux

    def changeResponse(self, rmf, arf, minenergy_keV, maxenergy_keV):
        """Create a fake spectrum using the response and use energy range given."""

        self.setPlaw(0.1, 1.7, 1)
        self.tempoutput = '/tmp/mbproj2d_temp_%i.fak' % os.getpid()
        deleteFile(self.tempoutput)
        self.write('data none\n')
        self.write('fakeit none & %s & %s & y & foo & %s & 1.0\n' %
            (rmf, arf, self.tempoutput))

        # this is the range we are interested in getting rates for
        self.write('ignore **:**-%f,%f-**\n' % (minenergy_keV, maxenergy_keV))
        self.throwAwayOutput()

    def dummyResponse(self):
        """Make a wide-energy band dummy response."""
        self.write('data none\n')
        self.write('dummyrsp 0.01 100. 10000\n')
        self.throwAwayOutput()

    def setPlaw(self, NH_1022pcm2, gamma, norm):
        """Set an absorbed powerlaw model."""
        self.write(
            'model TBabs(powerlaw) & %g & %g & %g\n' %
            (NH_1022pcm2, gamma, norm)
        )
        self.throwAwayOutput()

    def setApec(self, NH_1022pcm2, T_keV, Z_solar, redshift, norm):
        """Set an absorbed apec model."""
        self.write(
            'model TBabs(apec) & %g & %g & %g & %g & %g\n' %
            (NH_1022pcm2, T_keV, Z_solar, redshift, norm)
        )
        self.throwAwayOutput()

    def finish(self):
        self.write('tclexit\n')
        #self.xspecsub.stdin.flush()
        self.throwAwayOutput()
        self.xspecsub.stdout.close()
        self.xspecsub.wait()
        if self.tempoutput:
            deleteFile(self.tempoutput)
        del _finishatexit[ _finishatexit.index(self) ]

class XSpecContext:
    """Manager to handle creating xspec instance and cleaning up."""

    def __enter__(self):
        self.xspec = XSpecHelper()
        return self.xspec

    def __exit__(self, ttype, value, tb):
        self.xspec.finish()

def _finishXSpecs():
    """Finish any remaining xspecs if finish() does not get called above."""
    while _finishatexit:
        _finishatexit[0].finish()

atexit.register(_finishXSpecs)

# multiprocessing doesn't call atexit unless we do this
def sigterm(num, frame):
    _finishXSpecs()
    sys.exit()
signal.signal(signal.SIGTERM, sigterm)
