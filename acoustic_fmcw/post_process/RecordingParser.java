import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class RecordingParser {
    private String filePath;

    public ArrayList<Data> getDataNodes() {
        return dataNodes;
    }

    private ArrayList<Data> dataNodes;
    public RecordingParser(String fileName) {
        this.filePath = fileName;
        dataNodes = new ArrayList<>();
    }

    public void parseRecording() {
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader(filePath));
            String line = null;
            ArrayList<String> temp = new ArrayList<>();
            int counter = 0;

            while (true) {
                line = reader.readLine();
                if(line == null) break;
                temp.add(line);
                counter += 1;
                if(counter % 4 == 0) {
                    dataNodes.add(
                            new Data(temp.get(0), temp.get(1), temp.get(2), temp.get(3))
                    );
                    temp.clear();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
