//Your implementation goes here
import java.io.*;
import java.util.*;

public class inspect {
	public static void main(String[] args) throws Exception {
		
        if (args.length != 1) {
            System.err.println("Incorrect Arguments");
            System.exit(-1);
        }

        // Read train
        File inFile = new File(args[0]);
        
        // If file doesnt exists, then create it
        if (!inFile.exists()) {
            System.err.println("No file called: " + args[0]);
            System.exit(-1);
        }

		BufferedReader br = null;
		StringBuilder sb = new StringBuilder();

    	// Read string from the input file
        String sCurrentLine;
        
        br = new BufferedReader(new FileReader(inFile));

        br.readLine();
        while ((sCurrentLine = br.readLine()) != null) {
            sb.append(sCurrentLine + "!");
    	}

        // String preprocessing
        String s = sb.toString();

        String[] ss = s.split("!");
        int row = ss.length;
        s = s.replaceAll("!", ",");
        ss = s.split(",");
        int col = ss.length / row;
        
        String[][] sss = new String[row][col];

        // Store the input into 2D matrix
        for (int i = 0; i < ss.length; i++) {
            sss[i/col][i%col] = ss[i];
        }

        sss = dataNormalization(sss);

        int num_pos = 0, num_neg = 0;
        for (int i = 0; i < row; i++) {
            if (sss[i][col-1].equals("1")) {
                num_pos++;
            } else {
                num_neg++;
            }
        }

        // System.out.println("num_pos: " + num_pos);
        // System.out.println("num_neg: " + num_neg);

        double p = num_pos * 1.0 / (num_pos + num_neg);
        double log2 = Math.log(2);
        double entropy = p * Math.log(1.0/p) / log2 + (1.0-p) * Math.log(1.0/(1.0-p)) / log2;

        double rate_error = Math.min(p, 1-p);

        System.out.println("enropy: " + entropy);
        System.out.println("error: "  + rate_error);
	}


    private static String[][] dataNormalization(String[][] sss) {

        List<String> attr_pos = new ArrayList<String>();

        int row = sss.length;
        int col = sss[0].length;

        attr_pos.add("high");
        attr_pos.add("expensive");
        attr_pos.add("large");
        attr_pos.add("fast");
        attr_pos.add("before1950");
        attr_pos.add("morethan3min");
        attr_pos.add("democrat");
        attr_pos.add("Two");
        attr_pos.add("A");
        attr_pos.add("yes");
        attr_pos.add("y");

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (attr_pos.contains(sss[i][j])) {
                    sss[i][j] = "1";
                } else {
                    sss[i][j] = "0";
                }
            }
        }

        return sss;
    }
}