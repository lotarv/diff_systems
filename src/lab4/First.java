package lab4;

import mpi.*;

import java.util.ArrayList;
import java.util.List;

public class First {
    private static final int[] VECTOR_SIZES = {100, 1000, 1000, 1000000};
    private static final int TAG_LENGTH = 100;
    private static final int TAG_VECTOR_A = 101;
    private static final int TAG_VECTOR_B = 102;
    private static final int TAG_RESULT = 200;

    private enum ExchangeMode {
        BLOCKING,
        SYNCHRONOUS,
        NONBLOCKING
    }

    public static void main(String[] args) {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        for (ExchangeMode mode : ExchangeMode.values()) {
            for (int vectorSize : VECTOR_SIZES) {
                MPI.COMM_WORLD.Barrier();
                runExperiment(mode, vectorSize, rank, size);
                MPI.COMM_WORLD.Barrier();
            }
        }

        MPI.Finalize();
    }

    private static void runExperiment(ExchangeMode mode, int vectorSize, int rank, int size) {
        int workerCount = size - 1;
        if (rank == 0 && workerCount <= 0) {
            double[] a = buildVector(vectorSize, 1.0);
            double[] b = buildVector(vectorSize, 0.5);
            long start = System.currentTimeMillis();
            double dot = computeLocalDot(a, b, 0, vectorSize);
            long elapsed = System.currentTimeMillis() - start;
            System.out.println("[mode=" + mode + ", size=" + vectorSize + "] dot=" + dot +
                    " time=" + elapsed + " ms (single process)");
            return;
        }

        if (rank == 0) {
            masterRoutine(mode, vectorSize, workerCount);
        } else {
            workerRoutine(mode);
        }
    }

    private static void masterRoutine(ExchangeMode mode, int vectorSize, int workerCount) {
        double[] vectorA = buildVector(vectorSize, 1.0);
        double[] vectorB = buildVector(vectorSize, 0.5);
        int baseChunk = workerCount == 0 ? vectorSize : vectorSize / workerCount;
        int remainder = workerCount == 0 ? 0 : vectorSize % workerCount;
        int offset = 0;

        List<Request> pendingSends = new ArrayList<>();
        long startTime = System.currentTimeMillis();

        for (int worker = 1; worker <= workerCount; worker++) {
            int chunk = baseChunk + (worker <= remainder ? 1 : 0);

            int[] meta = {chunk};
            pendingSends.add(sendInts(meta, 0, 1, worker, TAG_LENGTH, mode));

            if (chunk > 0) {
                pendingSends.add(sendDoubles(vectorA, offset, chunk, worker, TAG_VECTOR_A, mode));
                pendingSends.add(sendDoubles(vectorB, offset, chunk, worker, TAG_VECTOR_B, mode));
            }

            offset += chunk;
        }

        waitRequests(pendingSends);

        double total = 0;
        double[] buf = new double[1];
        for (int worker = 1; worker <= workerCount; worker++) {
            MPI.COMM_WORLD.Recv(buf, 0, 1, MPI.DOUBLE, worker, TAG_RESULT);
            total += buf[0];
        }

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println("[mode=" + mode + ", size=" + vectorSize + "] dot=" + total +
                " time=" + elapsed + " ms");
    }

    private static void workerRoutine(ExchangeMode mode) {
        int[] meta = new int[1];
        receiveInts(meta, 0, 1, 0, TAG_LENGTH, mode);
        int chunk = meta[0];

        double[] partA = new double[chunk];
        double[] partB = new double[chunk];

        List<Request> pendingReceives = new ArrayList<>();

        if (chunk > 0) {
            receiveDoubles(partA, 0, chunk, 0, TAG_VECTOR_A, mode, pendingReceives);
            receiveDoubles(partB, 0, chunk, 0, TAG_VECTOR_B, mode, pendingReceives);
        }

        waitRequests(pendingReceives);

        double localResult = chunk == 0 ? 0 : computeLocalDot(partA, partB, 0, chunk);
        double[] resultBuf = {localResult};
        Request resultRequest = sendDoubles(resultBuf, 0, 1, 0, TAG_RESULT, mode);
        waitRequest(resultRequest);
    }

    private static Request sendInts(int[] buf, int offset, int count, int dest, int tag, ExchangeMode mode) {
        if (mode == ExchangeMode.NONBLOCKING) {
            return MPI.COMM_WORLD.Isend(buf, offset, count, MPI.INT, dest, tag);
        } else if (mode == ExchangeMode.SYNCHRONOUS) {
            MPI.COMM_WORLD.Ssend(buf, offset, count, MPI.INT, dest, tag);
        } else {
            MPI.COMM_WORLD.Send(buf, offset, count, MPI.INT, dest, tag);
        }
        return null;
    }

    private static Request sendDoubles(double[] buf, int offset, int count, int dest, int tag, ExchangeMode mode) {
        if (count == 0) {
            return null;
        }

        if (mode == ExchangeMode.NONBLOCKING) {
            return MPI.COMM_WORLD.Isend(buf, offset, count, MPI.DOUBLE, dest, tag);
        } else if (mode == ExchangeMode.SYNCHRONOUS) {
            MPI.COMM_WORLD.Ssend(buf, offset, count, MPI.DOUBLE, dest, tag);
        } else {
            MPI.COMM_WORLD.Send(buf, offset, count, MPI.DOUBLE, dest, tag);
        }
        return null;
    }

    private static void receiveInts(int[] buf, int offset, int count, int source, int tag, ExchangeMode mode) {
        if (mode == ExchangeMode.NONBLOCKING) {
            Request req = MPI.COMM_WORLD.Irecv(buf, offset, count, MPI.INT, source, tag);
            req.Wait();
            return;
        }

        MPI.COMM_WORLD.Recv(buf, offset, count, MPI.INT, source, tag);
    }

    private static void receiveDoubles(double[] buf, int offset, int count, int source, int tag,
                                       ExchangeMode mode, List<Request> pending) {
        if (count == 0) {
            return;
        }

        if (mode == ExchangeMode.NONBLOCKING) {
            pending.add(MPI.COMM_WORLD.Irecv(buf, offset, count, MPI.DOUBLE, source, tag));
        } else {
            MPI.COMM_WORLD.Recv(buf, offset, count, MPI.DOUBLE, source, tag);
        }
    }

    private static void waitRequests(List<Request> requests) {
        int total = 0;
        for (Request request : requests) {
            if (request != null) {
                total++;
            }
        }

        if (total == 0) {
            return;
        }

        Request[] arr = new Request[total];
        int idx = 0;
        for (Request request : requests) {
            if (request != null) {
                arr[idx++] = request;
            }
        }
        Request.Waitall(arr);
    }

    private static void waitRequest(Request request) {
        if (request != null) {
            request.Wait();
        }
    }

    private static double[] buildVector(int size, double factor) {
        double[] vector = new double[size];
        for (int i = 0; i < size; i++) {
            vector[i] = factor * (i + 1);
        }
        return vector;
    }

    private static double computeLocalDot(double[] a, double[] b, int offset, int length) {
        double sum = 0;
        for (int i = 0; i < length; i++) {
            sum += a[offset + i] * b[offset + i];
        }
        return sum;
    }
}
