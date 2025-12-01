package lab3;
import mpi.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Second {
    public static void main(String[] args) {
        int level2BlocksCount = 6;
        long globalStartTime = System.currentTimeMillis();

        MPI.Init(args);
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();
        int dataTag = 0;
        int timeTag = 999;

        int minBlockCount = level2BlocksCount + 1 + 1;
        if (size < minBlockCount) {
            if (rank == 0) {
                System.out.println("Not enough processes! Need at least " + minBlockCount +
                        ", executed:" + size);
            }
            MPI.Finalize();
            return;
        }

        if (rank > level2BlocksCount) {
            int number = (int) (Math.random() * 100);
            int[] data = {number};
            int dest = (rank % level2BlocksCount) + 1;

            MPI.COMM_WORLD.Send(data, 0, 1, MPI.INT, dest, dataTag);
            System.out.println("Process " + rank + " sent " + number + " → process " + dest);
        }
        else if (rank > 0) {
            long blockStartTime = System.currentTimeMillis();

            int sendersCount = 0;
            for (int sender = level2BlocksCount + 1; sender < size; sender++) {
                if ((sender % level2BlocksCount) + 1 == rank) sendersCount++;
            }

            int[][] bufs = new int[sendersCount][1];
            Request[] requests = new Request[sendersCount];

            for (int i = 0; i < sendersCount; i++) {
                requests[i] = MPI.COMM_WORLD.Irecv(bufs[i], 0, 1, MPI.INT, MPI.ANY_SOURCE, dataTag);
            }
            Request.Waitall(requests);

            int[] collected = new int[sendersCount];
            for (int i = 0; i < sendersCount; i++) {
                collected[i] = bufs[i][0];
            }
            Arrays.sort(collected);
            System.out.println("Process " + rank + " collected and sorted: " + Arrays.toString(collected));

            MPI.COMM_WORLD.Send(collected, 0, collected.length, MPI.INT, 0, dataTag);

            long blockTime = System.currentTimeMillis() - blockStartTime;
            System.out.println("Process " + rank + " finished block in " + blockTime + " ms");

            double[] timeData = {blockTime};
            MPI.COMM_WORLD.Send(timeData, 0, 1, MPI.DOUBLE, 0, timeTag);
        }
        else { // rank == 0
            long blockStartTime = System.currentTimeMillis();
            List<Integer> finalResult = new ArrayList<>();
            double[] blockTimes = new double[level2BlocksCount + 1];

            for (int sender = 1; sender <= level2BlocksCount; sender++) {
                Status st = MPI.COMM_WORLD.Probe(sender, dataTag);
                int count = st.Get_count(MPI.INT);
                int[] buf = new int[count];
                MPI.COMM_WORLD.Recv(buf, 0, count, MPI.INT, sender, dataTag);
                System.out.println("Process 0 received from " + sender + ": " + Arrays.toString(buf));
                for (int x : buf) finalResult.add(x);
            }

            for (int sender = 1; sender <= level2BlocksCount; sender++) {
                double[] timeBuf = new double[1];
                MPI.COMM_WORLD.Recv(timeBuf, 0, 1, MPI.DOUBLE, sender, timeTag);
                blockTimes[sender] = timeBuf[0];
            }
            blockTimes[0] = System.currentTimeMillis() - blockStartTime;

            Collections.sort(finalResult);

            System.out.println("\n=== FINAL RESULT ===");
            System.out.println("Result: " + finalResult);

            long totalTime = System.currentTimeMillis() - globalStartTime;

            System.out.println("\n=== Relative processing time ===");
            for (int r = 0; r <= level2BlocksCount; r++) {
                double percent = blockTimes[r] * 100.0 / totalTime;
                System.out.println("Process " + r + " worked " + (long)blockTimes[r] +
                        " ms (" + String.format("%.1f", percent) + "% of total)");
            }
            System.out.println("Program finished in " + totalTime + " ms");
        }

        MPI.Finalize();
    }
}