package lab4;

import mpi.Intracomm;
import mpi.MPI;

public class Second {
    private static final int[][] MATRIX_DIMENSIONS = {
            {120, 80, 100},
            {200, 150, 180},
            {256, 256, 256}
    };

    private static final int TAG_A_BLOCK = 500;
    private static final int TAG_B_BLOCK = 501;
    private static final int TAG_RESULT_BLOCK = 502;

    public static void main(String[] args) {
        MPI.Init(args);

        Intracomm world = MPI.COMM_WORLD;
        int rank = world.Rank();
        int size = world.Size();

        int[] grid = computeGrid(size);
        int gridRows = grid[0];
        int gridCols = grid[1];

        int rowCoord = rank / gridCols;
        int colCoord = rank % gridCols;

        Intracomm rowComm = world.Split(rowCoord, colCoord);
        Intracomm colComm = world.Split(gridRows + colCoord, rowCoord);

        for (int[] dims : MATRIX_DIMENSIONS) {
            world.Barrier();
            runExperiment(dims, rank, gridRows, gridCols, rowCoord, colCoord, rowComm, colComm);
            world.Barrier();
        }

        MPI.Finalize();
    }

    private static void runExperiment(int[] dims, int rank, int gridRows, int gridCols,
                                      int rowCoord, int colCoord, Intracomm rowComm, Intracomm colComm) {
        int rowsA = dims[0];
        int shared = dims[1];
        int colsB = dims[2];

        int[] rowCounts = partition(rowsA, gridRows);
        int[] rowOffsets = offsets(rowCounts);
        int[] colCounts = partition(colsB, gridCols);
        int[] colOffsets = offsets(colCounts);

        int localRows = rowCounts[rowCoord];
        int localCols = colCounts[colCoord];

        double[] fullA = null;
        double[] fullB = null;
        double[] fullC = null;

        if (rank == 0) {
            fullA = buildMatrix(rowsA, shared, 1.0);
            fullB = buildMatrix(shared, colsB, 0.5);
            fullC = new double[rowsA * colsB];
        }

        double[] rowBlock = new double[localRows * shared];
        double[] colBlock = new double[shared * localCols];

        long start = System.currentTimeMillis();

        distributeRowStrips(rank, fullA, shared, rowCounts, rowOffsets, gridCols,
                rowCoord, colCoord, rowBlock, rowComm);
        distributeColStrips(rank, fullB, shared, colsB, colCounts, colOffsets, rowCoord,
                colCoord, colBlock, colComm);

        double[] localC = multiplyBlocks(rowBlock, colBlock, localRows, shared, localCols);

        gatherResult(rank, localC, rowCounts, rowOffsets, colCounts, colOffsets,
                rowCoord, colCoord, gridCols, rowsA, colsB, fullC);

        long elapsed = System.currentTimeMillis() - start;

        if (rank == 0) {
            double[] reference = multiplySequential(fullA, fullB, rowsA, shared, colsB);
            double maxDiff = maxDifference(fullC, reference);
            System.out.println("[dims=" + rowsA + "x" + shared + " * " + shared + "x" + colsB +
                    ", grid=" + gridRows + "x" + gridCols + "] time=" + elapsed +
                    " ms, maxDiff=" + maxDiff);
        }
    }

    private static void distributeRowStrips(int rank, double[] fullA, int shared, int[] rowCounts,
                                            int[] rowOffsets, int gridCols, int rowCoord, int colCoord,
                                            double[] rowBlock, Intracomm rowComm) {
        if (colCoord == 0) {
            if (rank == 0) {
                for (int r = 0; r < rowCounts.length; r++) {
                    int rows = rowCounts[r];
                    int dest = r * gridCols;
                    double[] temp = new double[rows * shared];
                    if (rows > 0) {
                        copyRows(fullA, shared, rowOffsets[r], rows, temp);
                    }

                    if (dest == 0) {
                        System.arraycopy(temp, 0, rowBlock, 0, temp.length);
                    } else {
                        MPI.COMM_WORLD.Send(temp, 0, temp.length, MPI.DOUBLE, dest, TAG_A_BLOCK);
                    }
                }
            } else {
                MPI.COMM_WORLD.Recv(rowBlock, 0, rowBlock.length, MPI.DOUBLE, 0, TAG_A_BLOCK);
            }
        }

        rowComm.Bcast(rowBlock, 0, rowBlock.length, MPI.DOUBLE, 0);
    }

    private static void distributeColStrips(int rank, double[] fullB, int shared, int totalCols,
                                            int[] colCounts, int[] colOffsets, int rowCoord, int colCoord,
                                            double[] colBlock, Intracomm colComm) {
        if (rowCoord == 0) {
            if (rank == 0) {
                for (int c = 0; c < colCounts.length; c++) {
                    int cols = colCounts[c];
                    int dest = c;
                    double[] temp = new double[shared * cols];
                    if (cols > 0) {
                        copyColumns(fullB, shared, totalCols, colOffsets[c], cols, temp);
                    }

                    if (dest == 0) {
                        System.arraycopy(temp, 0, colBlock, 0, temp.length);
                    } else {
                        MPI.COMM_WORLD.Send(temp, 0, temp.length, MPI.DOUBLE, dest, TAG_B_BLOCK);
                    }
                }
            } else {
                MPI.COMM_WORLD.Recv(colBlock, 0, colBlock.length, MPI.DOUBLE, 0, TAG_B_BLOCK);
            }
        }

        colComm.Bcast(colBlock, 0, colBlock.length, MPI.DOUBLE, 0);
    }

    private static void gatherResult(int rank, double[] localC, int[] rowCounts, int[] rowOffsets,
                                     int[] colCounts, int[] colOffsets, int rowCoord, int colCoord,
                                     int gridCols, int rowsA, int colsB, double[] fullC) {
        if (rank == 0) {
            insertBlock(fullC, rowsA, colsB, rowOffsets[rowCoord], colOffsets[colCoord],
                    rowCounts[rowCoord], colCounts[colCoord], localC);

            double[] buffer;
            for (int proc = 1; proc < MPI.COMM_WORLD.Size(); proc++) {
                int procRow = proc / gridCols;
                int procCol = proc % gridCols;
                int blockRows = rowCounts[procRow];
                int blockCols = colCounts[procCol];
                int blockSize = blockRows * blockCols;
                buffer = new double[blockSize];
                MPI.COMM_WORLD.Recv(buffer, 0, blockSize, MPI.DOUBLE, proc, TAG_RESULT_BLOCK);
                insertBlock(fullC, rowsA, colsB, rowOffsets[procRow], colOffsets[procCol],
                        blockRows, blockCols, buffer);
            }
        } else {
            MPI.COMM_WORLD.Send(localC, 0, localC.length, MPI.DOUBLE, 0, TAG_RESULT_BLOCK);
        }
    }

    private static int[] computeGrid(int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Process count must be positive");
        }

        int rows = (int) Math.floor(Math.sqrt(size));
        while (rows > 1 && size % rows != 0) {
            rows--;
        }
        int cols = size / rows;
        return new int[]{rows, cols};
    }

    private static int[] partition(int total, int parts) {
        int[] counts = new int[parts];
        if (parts == 0) {
            return counts;
        }

        int base = total / parts;
        int remainder = total % parts;
        for (int i = 0; i < parts; i++) {
            counts[i] = base + (i < remainder ? 1 : 0);
        }
        return counts;
    }

    private static int[] offsets(int[] counts) {
        int[] offsets = new int[counts.length];
        for (int i = 1; i < counts.length; i++) {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }
        return offsets;
    }

    private static double[] buildMatrix(int rows, int cols, double factor) {
        double[] matrix = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i * cols + j] = factor * (i + 1) + (j + 1) * 0.1;
            }
        }
        return matrix;
    }

    private static void copyRows(double[] src, int cols, int startRow, int rowCount, double[] dest) {
        if (rowCount == 0) {
            return;
        }
        int from = startRow * cols;
        System.arraycopy(src, from, dest, 0, rowCount * cols);
    }

    private static void copyColumns(double[] src, int rows, int totalCols, int startCol, int colCount,
                                    double[] dest) {
        for (int r = 0; r < rows; r++) {
            int srcOffset = r * totalCols + startCol;
            int destOffset = r * colCount;
            System.arraycopy(src, srcOffset, dest, destOffset, colCount);
        }
    }

    private static double[] multiplyBlocks(double[] rowBlock, double[] colBlock, int rowCount,
                                           int shared, int colCount) {
        double[] result = new double[rowCount * colCount];
        if (rowCount == 0 || colCount == 0) {
            return result;
        }

        for (int i = 0; i < rowCount; i++) {
            int rowOffset = i * shared;
            int resultOffset = i * colCount;
            for (int k = 0; k < shared; k++) {
                double aVal = rowBlock[rowOffset + k];
                int colOffset = k * colCount;
                for (int j = 0; j < colCount; j++) {
                    result[resultOffset + j] += aVal * colBlock[colOffset + j];
                }
            }
        }
        return result;
    }

    private static void insertBlock(double[] dest, int destRows, int destCols, int startRow, int startCol,
                                    int blockRows, int blockCols, double[] block) {
        for (int r = 0; r < blockRows; r++) {
            int destOffset = (startRow + r) * destCols + startCol;
            int srcOffset = r * blockCols;
            System.arraycopy(block, srcOffset, dest, destOffset, blockCols);
        }
    }

    private static double[] multiplySequential(double[] a, double[] b, int rowsA, int shared, int colsB) {
        double[] result = new double[rowsA * colsB];
        for (int i = 0; i < rowsA; i++) {
            for (int k = 0; k < shared; k++) {
                double aVal = a[i * shared + k];
                for (int j = 0; j < colsB; j++) {
                    result[i * colsB + j] += aVal * b[k * colsB + j];
                }
            }
        }
        return result;
    }

    private static double maxDifference(double[] actual, double[] expected) {
        double diff = 0;
        for (int i = 0; i < actual.length; i++) {
            diff = Math.max(diff, Math.abs(actual[i] - expected[i]));
        }
        return diff;
    }
}
