//Your implementation goes here
import java.io.*;
import java.util.*;

public class decisionTree {

    private static String[] attribute;
    private static String[][] attribute_val;

	public static void main(String[] args) throws Exception {

        if (args.length != 2) {
            System.err.println("Incorrect Arguments");
            System.exit(-1);
        }
		
        // Read training data
        String[][] sss = readData(args[0]);
        int row = sss.length;
        int col = sss[0].length;

        // Get the overall statistics of training data
        int num_data = 1 << col;
        int[] data = new int[num_data];
        int[] flag = new int[row];
        for (int i = 0; i < row; i++) {

            // Get the index of data
            int index = 0;
            int pow = 1;
            for (int j = col-1; j >= 0; j--) {
                index += Integer.parseInt(sss[i][j]) * pow;
                pow *= 2;
            }
            flag[i] = index;

            data[index]++;
        }

        // Train data
        // Count the number of positive and negative of each bit

        // Only need to use pos_num and neg_num to do further computation
        int[][] data_2d = new int[col-1][4]; // Store the result

        // Traverse each attribute
        for (int i = 0; i < col-1; i++) {
            // System.out.println("i: " + i);

            // For each attribute, traverse the data[]
            for (int j = 0; j < num_data; j++) {
                String temp = "" + Integer.toBinaryString(j);
                while (temp.length() < col) {
                    temp = "0" + temp;
                }
                // System.out.println("temp: " + temp + "  j: " + j + "  data[j]: " + data[j]);

                // Only for possibilities for temp[i] and temp[col-1]
                if (temp.charAt(i) == '0' && temp.charAt(col-1) == '0') {
                    data_2d[i][0] += data[j];
                    // System.out.println("data_2d[" + i + "][0]: " + data_2d[i][0]);
                } else if (temp.charAt(i) == '0' && temp.charAt(col-1) == '1') {
                    data_2d[i][1] += data[j];
                    // System.out.println("data_2d[" + i + "][1]: " + data_2d[i][1]);
                } else if (temp.charAt(i) == '1' && temp.charAt(col-1) == '0') {
                    data_2d[i][2] += data[j];
                    // System.out.println("data_2d[" + i + "][2]: " + data_2d[i][2]);
                } else if (temp.charAt(i) == '1' && temp.charAt(col-1) == '1') {
                    data_2d[i][3] += data[j];
                    // System.out.println("data_2d[" + i + "][3]: " + data_2d[i][3]);
                }
            }
            // System.out.println();
        }

        // for (int i = 0; i < 2; i++) {
        //     System.out.println("data_2d[" + i + "][0]: " + data_2d[i][0]);
        //     System.out.println("data_2d[" + i + "][1]: " + data_2d[i][1]);
        //     System.out.println("data_2d[" + i + "][2]: " + data_2d[i][2]);
        //     System.out.println("data_2d[" + i + "][3]: " + data_2d[i][3]);
        // }

        // Count number of positive result
        int num_pos = 0, num_neg = 0;
        for (int i = 0; i < row; i++) {
            if (sss[i][col-1].equals("1")) {
                num_pos++;
            } else {
                num_neg++;
            }
        }

        double log2 = Math.log(2);
        double rate_pos = num_pos * 1.0 / row;
        double e = rate_pos * (Math.log(1.0/rate_pos)/ log2) + (1-rate_pos) * (Math.log(1.0/(1-rate_pos)) / log2);
        // System.out.println(num_pos + "   " + rate_pos + "   " + e + "!!!");

        int max_index_1 = 0;
        double max_mutual_info_1 = -1.0;


        // Calculate Mutual Info to Choose root
        double[] mutual_info_1 = new double[col-1];

        // Store all mutual info
        double[][] mutual_info_res = new double[col-1][2];

        for (int i = 0; i < col-1; i++) {
            double entropy = 0.0;

            int temp_num = data_2d[i][0] + data_2d[i][1] + data_2d[i][2] + data_2d[i][3];
            if (temp_num == 0) {
                // For fear of meeting NaN
                continue;
            }
            for (int j = 0; j < 4; j += 2) {
                int temp2 = data_2d[i][j] + data_2d[i][j+1];
                double p  = temp2 * 1.0 / temp_num;
                double p1 = data_2d[i][j]   * 1.0 / temp2;
                double p2 = data_2d[i][j+1] * 1.0 / temp2;

                if (p1 != 0.0 && p2 != 0.0) {
                    mutual_info_res[i][j/2] += p1 * Math.log(1.0/p1) / log2;
                    mutual_info_res[i][j/2] += p2 * Math.log(1.0/p2) / log2;
                }

                entropy += mutual_info_res[i][j/2] * p;
                // System.out.println("mutual_info_res[" + i + "][" + (j/2) + "] = " + mutual_info_res[i][j/2]);
            }
            // System.out.println(entropy + "\n");
            mutual_info_1[i] = e - entropy;

            // Get the max muaual info and its index
            if (mutual_info_1[i] > max_mutual_info_1) {
                max_mutual_info_1 = mutual_info_1[i];
                max_index_1 = i;
            }
        }

        // Preprocessing data for layer 2
        // Only need to use pos_num and neg_num to do further computation
        int[][][] data_3d = new int[2][col-1][4]; // Store the result

        // Traverse each attribute
        for (int i = 0; i < col-1; i++) {

            // Skip first layer's data
            if (i == max_index_1) {
                data_3d[0][max_index_1][0] = -1;
                data_3d[0][max_index_1][1] = -1;
                data_3d[0][max_index_1][2] = -1;
                data_3d[0][max_index_1][3] = -1;
                data_3d[1][max_index_1][0] = -1;
                data_3d[1][max_index_1][1] = -1;
                data_3d[1][max_index_1][2] = -1;
                data_3d[1][max_index_1][3] = -1;
                continue;
            }

            // For each attribute, traverse the data[]
            for (int j = 0; j < num_data; j++) {
                String temp = "" + Integer.toBinaryString(j);
                while (temp.length() < col) {
                    temp = "0" + temp;
                }
                // System.out.println("temp: " + temp + "  data[" + j + "]: " + data[j]);

                int root = 0;
                if (temp.charAt(max_index_1) == '0') {
                    root = 0;
                } else {
                    root = 1;
                }

                // Only for possibilities for temp[i] and temp[col-1]
                if (temp.charAt(i) == '0' && temp.charAt(col-1) == '0') {
                    data_3d[root][i][0] += data[j];
                    // System.out.println("i: " + i + " data_2d[0]: " + data_2d[i][0]);
                } else if (temp.charAt(i) == '0' && temp.charAt(col-1) == '1') {
                    data_3d[root][i][1] += data[j];
                    // System.out.println("i: " + i + " data_2d[1]: " + data_2d[i][1]);
                } else if (temp.charAt(i) == '1' && temp.charAt(col-1) == '0') {
                    data_3d[root][i][2] += data[j];
                    // System.out.println("i: " + i + " data_2d[2]: " + data_2d[i][2]);
                } else if (temp.charAt(i) == '1' && temp.charAt(col-1) == '1') {
                    data_3d[root][i][3] += data[j];
                    // System.out.println("i: " + i + " data_2d[3]: " + data_2d[i][3]);
                }
            }
            // System.out.println();
        }

        // for (int i = 0; i < 2; i++) {
        //     for (int j = 0; j < col-1; j++) {
        //         System.out.println("data_3d[" + i + "][" + j + "][" + 0 + "]" + data_3d[i][j][0]);
        //         System.out.println("data_3d[" + i + "][" + j + "][" + 1 + "]" + data_3d[i][j][1]);
        //         System.out.println("data_3d[" + i + "][" + j + "][" + 2 + "]" + data_3d[i][j][2]);
        //         System.out.println("data_3d[" + i + "][" + j + "][" + 3 + "]" + data_3d[i][j][3]);
        //     }
        // }

        // Calculate Mutual Info to + of root
        double[][] mutual_info_2 = new double[2][col-1];
        double max_mutual_info_2_1 = Double.MIN_VALUE;
        int max_index_2_1 = 0;
        double e_1 = mutual_info_res[max_index_1][1];
        // System.out.println("+ e: " + e_1);


        if (e_1 >= 0.1) {

            for (int i = 0; i < col-1; i++) {
                // System.out.println("i: " + i);

                if (i == max_index_1) {
                    mutual_info_2[1][i] = Double.MIN_VALUE;
                    continue;
                }

                double entropy = 0.0;
                int temp_num = data_3d[1][i][0] + data_3d[1][i][1] + data_3d[1][i][2] + data_3d[1][i][3];
                if (temp_num == 0) {
                    // For fear of meeting NaN
                    continue;
                }
                // System.out.println("temp_num: " + temp_num);
                for (int j = 0; j < 4; j += 2) {
                    double temp2 = data_3d[1][i][j] + data_3d[1][i][j+1];
                    if (temp2 == 0.0) {
                        continue;
                    }

                    double p = temp2 * 1.0 / temp_num;
                    double p1 = data_3d[1][i][j] * 1.0 / temp2;
                    double p2 = data_3d[1][i][j+1] * 1.0 / temp2;
                    // System.out.println("p1: " + p1);
                    // System.out.println("p2: " + p2);

                    if (p1 != 0.0 && p2 != 0.0) {
                        entropy += p * (p1 * Math.log(1.0/p1) / log2);
                        entropy += p * (p2 * Math.log(1.0/p2) / log2);
                    }
                    // System.out.println("entropy: " + entropy);
                    // System.out.println(p+"()()()()()");
                }
                mutual_info_2[1][i] = e_1 - entropy;

                // System.out.println("mutual_info_2[1][" + i + "]: " + mutual_info_2[1][i]);
                // System.out.println("max_mutual_info_2_1: " + max_mutual_info_2_1);
                
                // Get the max muaual info and its index
                if (mutual_info_2[1][i] > max_mutual_info_2_1) {
                    // System.out.println("i: " + i + "Chagne max index now!");
                    max_mutual_info_2_1 = mutual_info_2[1][i];
                    max_index_2_1 = i;
                }
            }

        }
        // System.out.println("max_mutual_info_2_1: " + max_mutual_info_2_1);
        // System.out.println("max_index_2_1: " + max_index_2_1);

        // If mutual info too small, then ignore it
        if (max_mutual_info_2_1 < 0.1) {
            e_1 = 0.0;
        }

        // Calculate Mutual Info to - of root
        int max_index_2_2 = 0;
        double max_mutual_info_2_2 = Double.MIN_VALUE;
        double e_0 = mutual_info_res[max_index_1][0];
        // System.out.println("- e: " + e_0);

        if (e_0 >= 0.1) {

            for (int i = 0; i < col-1; i++) {
                // System.out.println("i: " + i);

                if (i == max_index_1) {
                    mutual_info_2[0][i] = Double.MIN_VALUE;
                    continue;
                }

                double entropy = 0.0;
                int temp_num = data_3d[0][i][0] + data_3d[0][i][1] + data_3d[0][i][2] + data_3d[0][i][3];
                if (temp_num == 0) {
                    // For fear of meeting NaN
                    continue;
                }

                for (int j = 0; j < 4; j += 2) {
                    double temp2 = data_3d[0][i][j] + data_3d[0][i][j+1];
                    if (temp2 == 0.0) {
                        continue;
                    }

                    double p = temp2 * 1.0 / temp_num;
                    double p1 = data_3d[0][i][j] * 1.0 / temp2;
                    double p2 = data_3d[0][i][j+1] * 1.0 / temp2;
                    // System.out.println("p1: " + p1);
                    // System.out.println("p2: " + p2);

                    if (p1 != 0.0 && p2 != 0.0) {
                        entropy += p * (p1 * Math.log(1.0/p1) / log2);
                        entropy += p * (p2 * Math.log(1.0/p2) / log2);
                    }
                    // System.out.println(p+"()()()()()");
                }
                mutual_info_2[0][i] = e_0 - entropy;

                // System.out.println("mutual_info_2[0][" + i + "]: " + mutual_info_2[0][i]);
                // System.out.println("max_mutual_info_2_2: " + max_mutual_info_2_2);

                // Get the max muaual info and its index
                if (mutual_info_2[0][i] > max_mutual_info_2_2) {
                    // System.out.println("i: " + i + "Chagne max index now!");
                    max_mutual_info_2_2 = mutual_info_2[0][i];
                    max_index_2_2 = i;
                }
            }
        }
        // System.out.println("max_mutual_info_2_2: " + max_mutual_info_2_2);
        // System.out.println("max_index_2_2: " + max_index_2_2);

        // If mutual info too small, then ignore it
        if (max_mutual_info_2_2 < 0.1) {
            e_0 = 0.0;
        }

        // Compute the overall result error rate for traininig data
        Double rate_error = computeError(sss, max_index_1, max_index_2_1, max_index_2_2, e_0, e_1, row, col, num_pos);

        // System.out.println();


        // Print result
        System.out.println("[" + num_pos + "+/" + num_neg + "-]");

        System.out.println(attribute[max_index_1] + " = " + attribute_val[max_index_1][0] + ": [" + data_2d[max_index_1][3] + "+/" + data_2d[max_index_1][2] + "-]");
        if (e_1 >= 0.1) {
            System.out.println("| " + attribute[max_index_2_1] + " = " + attribute_val[max_index_2_1][0] + ": [" + data_3d[1][max_index_2_1][3] + "+/" + data_3d[1][max_index_2_1][2] + "-]");
            System.out.println("| " + attribute[max_index_2_1] + " = " + attribute_val[max_index_2_1][1] + ": [" + data_3d[1][max_index_2_1][1] + "+/" + data_3d[1][max_index_2_1][0] + "-]");
        }

        System.out.println(attribute[max_index_1] + " = " + attribute_val[max_index_1][1] + ": [" + data_2d[max_index_1][1] + "+/" + data_2d[max_index_1][0] + "-]");
        if (e_0 >= 0.1) {
            System.out.println("| " + attribute[max_index_2_2] + " = " + attribute_val[max_index_2_2][0] + ": [" + data_3d[0][max_index_2_2][3] + "+/" + data_3d[0][max_index_2_2][2] + "-]");
            System.out.println("| " + attribute[max_index_2_2] + " = " + attribute_val[max_index_2_2][1] + ": [" + data_3d[0][max_index_2_2][1] + "+/" + data_3d[0][max_index_2_2][0] + "-]");
        }

        System.out.println("error(train): " + rate_error);

        // Read test data
        sss = readData(args[1]);
        row = sss.length;
        col = sss[0].length;

        // Count number of positive result
        num_pos = 0;
        for (int i = 0; i < row; i++) {
            if (sss[i][col-1].equals("1")) {
                num_pos++;
            }
        }

        // Compute the overall result error rate for test data
        rate_error = computeError(sss, max_index_1, max_index_2_1, max_index_2_2, e_0, e_1, row, col, num_pos);
        
        System.out.println("error(test): " + rate_error);
	}



    private static double computeError(String[][] sss, int max_index_1, int max_index_2_1, int max_index_2_2, double e_0, double e_1, int row, int col, int num_pos) {

        // Count the right answer in + branch
        int count_2_1_pos_0 = 0, count_2_1_pos_1 = 0, count_2_1_neg_0 = 0, count_2_1_neg_1 = 0;
        int count_2_1_0 = 0, count_2_1_1 = 0;
        if (e_1 >= 0.1) {
            for (int i = 0; i < row; i++) {
                if (sss[i][max_index_1].equals("0")) {
                    continue;
                }
                if (sss[i][max_index_2_1].equals("1")) {
                    // + branch
                    if (sss[i][col-1].equals("1")) {
                        count_2_1_pos_1++;
                    } else {
                        count_2_1_pos_0++;
                    }
                } else {
                    // - branch
                    if (sss[i][col-1].equals("1")) {
                        count_2_1_neg_1++;
                    } else {
                        count_2_1_neg_0++;
                    }
                }
            }
        } else {
            for (int i = 0; i < row; i++) {
                if (sss[i][max_index_1].equals("0")) {
                    continue;
                }
                if (sss[i][col-1].equals("1")) {
                    count_2_1_1++;
                } else {
                    count_2_1_0++;
                }
            }
        }

        // Count the right answer in - branch
        int count_2_2_0 = 0, count_2_2_1 = 0;
        int count_2_2_pos_0 = 0, count_2_2_pos_1 = 0, count_2_2_neg_0 = 0, count_2_2_neg_1 = 0;
        if (e_0 >= 0.1) {
            for (int i = 0; i < row; i++) {
                if (sss[i][max_index_1].equals("1")) {
                    continue;
                }
                if (sss[i][max_index_2_2].equals("1")) {
                    // + branch
                    if (sss[i][col-1].equals("1")) {
                        count_2_2_pos_1++;
                    } else {
                        count_2_2_pos_0++;
                    }
                } else {
                    // - branch
                    if (sss[i][col-1].equals("1")) {
                        count_2_2_neg_1++;
                    } else {
                        count_2_2_neg_0++;
                    }
                }
            }

        } else {
            for (int i = 0; i < row; i++) {
                if (sss[i][max_index_1].equals("1")) {
                    continue;
                }
                if (sss[i][col-1].equals("1")) {
                    count_2_2_1++;
                } else {
                    count_2_2_0++;
                }
            }
        }

        double rate_error = 0.0;
        
        if (e_0 < 0.1 && e_1 < 0.1) {
            // No branches in level 2
            double p = num_pos * 1.0 / row;
            rate_error = Math.min(p, 1-p);

        } else if (e_0 >= 0.1 && e_1 >= 0.1) {
            // Two branches in level 2
            int error = 0;
            
            error += Math.min(count_2_1_pos_0, count_2_1_pos_1);
            error += Math.min(count_2_1_neg_0, count_2_1_neg_1);
            error += Math.min(count_2_2_pos_0, count_2_2_pos_1);
            error += Math.min(count_2_2_neg_0, count_2_2_neg_1);
            rate_error = error * 1.0 / row;

        } else if (e_0 >= 0.1 && e_1 < 0.1) {
            int error = 0;

            error += Math.min(count_2_2_pos_0, count_2_2_pos_1);
            error += Math.min(count_2_2_neg_0, count_2_2_neg_1);
            error += Math.min(count_2_1_1, count_2_1_0);
            rate_error = error * 1.0 / row;

        } else {
            int error = 0;

            error += Math.min(count_2_1_pos_0, count_2_1_pos_1);
            error += Math.min(count_2_1_neg_0, count_2_1_neg_1);
            error += Math.min(count_2_2_1, count_2_2_0);
            rate_error = error * 1.0 / row;
        }

        return rate_error;
    }

    private static String[][] dataNormalization(String[][] sss) {

        // List<String> attr_neg = new ArrayList<String>();
        List<String> attr_pos = new ArrayList<String>();

        int row = sss.length;
        int col = sss[0].length;
        attribute_val = new String[col][2];

        // attr_neg.add("low");
        // attr_neg.add("cheap");
        // attr_neg.add("small");
        // attr_neg.add("slow");
        // attr_neg.add("after1950");
        // attr_neg.add("lessthan3min");
        // attr_neg.add("MoreThanTwo");
        // attr_neg.add("republican");
        // attr_neg.add("notA");
        // attr_neg.add("no");
        // attr_neg.add("n");

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
                    attribute_val[j][0] = sss[i][j];
                    sss[i][j] = "1";
                } else {
                    attribute_val[j][1] = sss[i][j];
                    sss[i][j] = "0";
                }
            }
        }


        // for (int i = 0; i < col; i++) {
        //     System.out.print(attribute_val[i][0] + " " + attribute_val[i][1] + "\t");
        // }
        // System.out.println();
        
        // s = s.replaceAll("low,", "0,");
        // s = s.replaceAll("cheap,", "0,");
        // s = s.replaceAll("small,", "0,");
        // s = s.replaceAll("slow,", "0,");
        // s = s.replaceAll("after1950,", "0,");
        // s = s.replaceAll("lessthan3min,", "0,");
        // s = s.replaceAll("MoreThanTwo,", "0,");
        // s = s.replaceAll("notA", "0");
        // s = s.replaceAll("n,", "0,");
        // s = s.replaceAll("no,", "0,");

        // s = s.replaceAll("high,", "1,");
        // s = s.replaceAll("expensive,", "1,");
        // s = s.replaceAll("large,", "1,");
        // s = s.replaceAll("fast,", "1,");
        // s = s.replaceAll("before1950,", "1,");
        // s = s.replaceAll("morethan3min,", "1,");
        // s = s.replaceAll("Two,", "1,");
        // s = s.replaceAll("A", "1");
        // s = s.replaceAll("y,", "1,");
        // s = s.replaceAll("yes,", "1,");

        // s = s.replaceAll("democrat", "1");
        // s = s.replaceAll("republican", "0");

        return sss;
    }



    private static String[][] readData(String fileName) throws Exception {

        // Read training data
        File inFile = new File(fileName);
        
        // If file doesnt exists, then create it
        if (!inFile.exists()) {
            System.err.println("No file called: " + fileName);
            System.exit(-1);
        }

        BufferedReader br = null;
        StringBuilder sb = new StringBuilder();

        // Read string from the input file
        String sCurrentLine;
        
        br = new BufferedReader(new FileReader(inFile));

        // Get the attributes and clean it
        String attr = br.readLine();
        attribute = attr.split(",");
        for (int i = 0; i < attribute.length; i++) {
            attribute[i] = attribute[i].trim();
        }
        // System.out.println(attribute[2]);

        while ((sCurrentLine = br.readLine()) != null) {
            //System.out.println(sCurrentLine);
            sb.append(sCurrentLine + "!");
            // Use . to mark the end of each line
        }

        String s = sb.toString();

        // Data Normalization
        //s = dataNormalization(s);

        String[] ss = s.split("!");
        // for (int i = 0; i < ss.length; i++) {
        //     System.out.println("i: " + i + "  " + ss[i]);
        // }

        // Get the number of columns
        int row = ss.length;
        // System.out.println(row);

        s = s.replaceAll("!", ",");
        ss = s.split(",");
        int col = ss.length / row;
        // System.out.println(col);
        // System.out.println(s);
        // System.out.println(ss.length);

        // // Change array to 2D matrix
        String[][] sss = new String[row][col];

        for (int i = 0; i < ss.length; i++) {
            sss[i/col][i%col] = ss[i];
        }

        sss = dataNormalization(sss);

        // for (int i = 0; i < row; i++) {
        //     for (int j = 0; j < col; j++) {
        //         System.out.print(sss[i][j] + "\t");
        //     }
        //     System.out.println(i);
        // }


        return sss;
    }
}