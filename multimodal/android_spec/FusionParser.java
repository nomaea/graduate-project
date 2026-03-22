// package android_spec;

// import com.google.gson.Gson;
// import com.google.gson.GsonBuilder;

public class FusionParser {

    private static final Gson gson = new GsonBuilder()
            .serializeNulls()
            .create();

    public static FusionResponse parse(String jsonStr) {
        return gson.fromJson(jsonStr, FusionResponse.class);
    }

    // 로컬 테스트용
    public static void main(String[] args) {
        String jsonStr = "... 여기에 sample_json.txt 내용 붙이기 ...";

        FusionResponse fr = parse(jsonStr);
        System.out.println("dominant = " + fr.fusion.dominant);
        System.out.println("safe     = " + fr.fusion.safe);
        System.out.println("stressed = " + fr.fusion.stressed);
        System.out.println("alert    = " + fr.alert.level);
    }
}

