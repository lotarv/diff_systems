package lab2;
import mpi.*;

public class Block {
    public static void main(String[] args) {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        int[] buf = new int[1];
        buf[0] = rank;
        int s = buf[0];

        int right = (rank + 1) % size;
        int left  = (rank - 1 + size) % size;

        if (rank == 0) {
            MPI.COMM_WORLD.Send(buf, 0, 1, MPI.INT, right, 0);
            System.out.println("Process " + rank + " send " + buf[0] + " to process " + right);

            MPI.COMM_WORLD.Recv(buf, 0, 1, MPI.INT, left, 0);
            System.out.println("Process " + rank + " received " + buf[0] + " from process " + left);
            s += buf[0];
            System.out.println("Final sum at P0 = " + s);
        } else {
            MPI.COMM_WORLD.Recv(buf, 0, 1, MPI.INT, left, 0);
            System.out.println("Process " + rank + " received " + buf[0] + " from process " + left);

            s += buf[0];
            System.out.println("Current s in process " + rank + " = " + s);

            buf[0] = s;
            MPI.COMM_WORLD.Send(buf, 0, 1, MPI.INT, right, 0);
            System.out.println("Process " + rank + " send " + buf[0] + " to process " + right);
        }

        MPI.Finalize();
    }
}
