# Copyright (C) 2020 Jeremy Sanders <jeremy@jeremysanders.net>
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

"""Module to run a forked parallel process for evalulating functions,
one at a time (ForkParallel) or using a queue of input data
(ForkQueue).
"""

import os
import socket
import struct
import select
import signal
import pickle

# special exit code to break out of child
exitcode = b'*[EX!T}*FORK'

# type used to send object size
sizesize = struct.calcsize('L')

def recvLen(sock, length):
    """Receive exactly length bytes from socket."""
    retn = b''
    while len(retn) < length:
        retn += sock.recv(length-len(retn))
    return retn

def sendItem(sock, item):
    """Pickle and send item to socket using size + pickled protocol."""
    pickled = pickle.dumps(item, -1)
    size = struct.pack('L', len(pickled))
    sock.sendall(size + pickled)

def recvItem(sock):
    """Receive pickled item from socket."""
    retn = sock.recv(64*1024)

    size = struct.unpack('L', retn[:sizesize])[0]
    retn = retn[sizesize:]

    while len(retn) < size:
        retn += sock.recv(size-len(retn))

    return pickle.loads(retn)

class ForkBase:
    """Base class for forking workers."""

    def __init__(self, func):
        self.func = func
        self.sock = None
        self.amparent = False

    def childLoop(self):
        """Wait for commands on the socket and execute."""

        if self.amparent:
            raise RuntimeError('Not child, or not started')

        # ignore ctrl+c
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # repeat until exit code or socket breaks
        try:
            while True:
                # get data to process
                args = recvItem(self.sock)

                #print('received', args)

                # exit if parent is done
                if args == exitcode:
                    break

                retn = []
                # presumably no socket error in below
                try:
                    # iterate over input and add result with index key
                    for arg in args:
                        res = self.func(arg)
                        retn.append(res)
                except Exception as e:
                    # send back an exception
                    retn = e

                # send back the result
                sendItem(self.sock, retn)

        except socket.error:
            #print('Breaking on socket error')
            pass

        #print('Exiting child')
        os._exit(os.EX_OK)

class ForkParallel(ForkBase):
    """Execute function in remote forked process."""

    def __init__(self, func):
        """Parallel forked runner for running func."""

        ForkBase.__init__(self, func)
        self.running = False

        # sockets communicate between forked processes
        parentsock, childsock = socket.socketpair()

        pid = os.fork()
        self.amparent = pid != 0

        if self.amparent:
            self.sock = parentsock
            childsock.close()
            self.pid = pid
        else:
            self.sock = childsock
            parentsock.close()
            self.childLoop()
            self.pid = None

    def __del__(self):
        """Tell child to close and close sockets."""

        if self.sock is not None:
            try:
                if self.amparent:
                    sendItem(self.sock, exitcode)
                self.sock.close()
            except socket.error:
                pass
        if self.pid is not None:
            # this avoids a zombie process
            os.waitpid(self.pid, 0)

    def send(self, args):
        """Send data to be processed."""

        if not self.amparent:
            raise RuntimeError('Not parent, or not started')
        if self.running:
            raise RuntimeError('Remote process is still executing')

        self.running = True
        sendItem(self.sock, [args])

    def query(self, timeout=0):
        """Return isdone,result from remote process."""

        if not self.amparent:
            raise RuntimeError('Not parent, or not started')
        if not self.running:
            raise RuntimeError('Remote process is already done')

        readsock, writesock, errsock = select.select(
            [self.sock], [], [], timeout)
        if readsock:
            retn = recvItem(self.sock)
            self.running = False
            if isinstance(retn, Exception):
                raise retn
            return True, retn[0]
        else:
            return False, None

    def wait(self):
        """Wait until a response, and return value."""
        while True:
            done, res = self.query(timeout=6000)
            if done:
                return res

class ForkQueue(ForkBase):
    """Execute function in multiple forked processes."""

    def __init__(self, func, instances, initfunc=None):
        """Initialise queue for func and with number of instances given.

        if initfunc is set, run this at first
        """

        ForkBase.__init__(self, func)

        self.socks = []
        self.pids = []

        for i in range(instances):
            parentsock, childsock = socket.socketpair()

            pid = os.fork()
            if pid == 0:
                # child process
                parentsock.close()
                self.sock = childsock
                self.amparent = False

                # close other children (we don't need to talk to them)
                del self.socks

                # call the initialise function, if required
                if initfunc is not None:
                    initfunc()

                # wait for commands from parent
                self.childLoop()

                # return here, or we get a fork bomb!
                return

            else:
                # parent process - keep track of children
                self.socks.append(parentsock)
                self.pids.append(pid)
                childsock.close()

        self.amparent = True

    def __del__(self):
        """Close child forks and close sockets."""
        if self.amparent:
            for sock in self.socks:
                try:
                    sendItem(sock, exitcode)
                    sock.close()
                except socket.error:
                    pass
            # this avoids zombie processes
            for pid in self.pids:
                os.waitpid(pid, 0)
        else:
            try:
                self.sock.close()
            except socket.error:
                pass

    def execute(self, argslist):
        """Execute the list of items on the queue.

        This version cheats by just splitting the input up into
        equal-sized chunks.

        Maybe the chunksize should be smaller, if some chunks would
        take much less time
        """

        if not self.amparent:
            raise RuntimeError('Not parent, or not started')

        # calculate number sent to each sock (making sure that the
        # number of items is <= than the number of sockets
        argslist = list(argslist)
        numargs = len(argslist)

        if numargs < len(self.socks):
            socks = self.socks[:numargs]
        else:
            socks = self.socks

        # round up chunk size
        chunksize = -(-numargs//len(socks))

        # send chunks to each forked process
        sockchunks = {}
        for idx, sock in enumerate(socks):
            sendItem(sock, argslist[idx*chunksize:(idx+1)*chunksize])
            sockchunks[sock] = idx

        # wait and collect responses
        retn = [None]*numargs
        while sockchunks:
            read, write, err = select.select(list(sockchunks), [], [])
            for sock in read:
                res = recvItem(sock)
                idx = sockchunks[sock]
                retn[idx*chunksize:(idx+1)*chunksize] = res
                del sockchunks[sock]

        return retn
