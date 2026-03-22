package android_spec;

import java.util.Map;

public class FusionResponse {
    public double timestamp;
    public FusionPart fusion;
    public FerPart fer;
    public SensorPart sensor;
    public AlertPart alert;
}

class FusionPart {
    public double safe;
    public double stressed;
    public String dominant;
    public double confidence;
}

class FerPart {
    public Map<String, Double> emotion_scores;
    public Map<String, Double> drowsy_scores;
}

class SensorPart {
    public Map<String, Double> raw_metrics;
    public Map<String, Double> emotion_scores;
}

class AlertPart {
    public String level;
    public String reason;
}
