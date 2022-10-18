import java.util.ArrayList;

public class Data {
    private String DateTime;
    private ArrayList<Double> chirp;
    private ArrayList<Double> record;
    private ArrayList<Double> direct;

    public Data(String dateTime, String chirp, String record, String direct) {
        DateTime = dateTime;
        this.chirp = parser(chirp);
        this.record = parser(record);
        this.direct = parser(direct);
    }

    public ArrayList<Double> parser(String stringArray) {
        stringArray = stringArray.split(":")[1];
        stringArray.trim();
        stringArray = stringArray.substring(1);
        stringArray = stringArray.substring(0, stringArray.length()-1);
        ArrayList<Double> result = new ArrayList<>();
        for(String item: stringArray.split(",")) {
            result.add(Double.parseDouble(item));
        }
        return result;
    }

    public String getDateTime() {
        return DateTime;
    }

    public void setDateTime(String dateTime) {
        DateTime = dateTime;
    }
    public ArrayList<Double> getChirp() {
        return chirp;
    }

    public void setChirp(ArrayList<Double> chirp) {
        this.chirp = chirp;
    }

    public ArrayList<Double> getRecord() {
        return record;
    }

    public void setRecord(ArrayList<Double> record) {
        this.record = record;
    }

    public ArrayList<Double> getDirect() {
        return direct;
    }

    public void setDirect(ArrayList<Double> direct) {
        this.direct = direct;
    }
}
