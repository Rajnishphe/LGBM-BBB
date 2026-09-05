import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { AlertCircle, Download, Loader2, BarChart3 } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { apiClient } from "@/lib/apiClient";

export function BatchPrediction() {
  const [moleculesText, setMoleculesText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const { toast } = useToast();

  const handleBatchPredict = async () => {
    if (!moleculesText.trim()) {
      toast({
        title: "Error",
        description: "Please enter molecule data",
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      // Parse input (SMILES,Name or just SMILES)
      const lines = moleculesText.trim().split('\n').filter(line => line.trim());
      const molecules = lines.map(line => {
        const parts = line.split(',');
        return {
          smiles: parts[0].trim(),
          name: parts[1]?.trim() || `Compound ${lines.indexOf(line) + 1}`
        };
      });

      const result = await apiClient.predictBatch(molecules, 0.5228);
      
      // Calculate confidence distribution client-side
      if (result.predictions) {
        const successfulPredictions = result.predictions.filter(p => p.status === "Success");
        const highConf = successfulPredictions.filter(p => p.confidence >= 0.8).length;
        const mediumConf = successfulPredictions.filter(p => p.confidence >= 0.6 && p.confidence < 0.8).length;
        const lowConf = successfulPredictions.filter(p => p.confidence < 0.6).length;
        
        // Ensure summary object exists
        if (!result.summary) {
          result.summary = {};
        }
        result.summary.high_confidence = highConf;
        result.summary.medium_confidence = mediumConf;
        result.summary.low_confidence = lowConf;
      }
      setResults(result);
      toast({
        title: "Batch Prediction Complete",
        description: `Processed ${result.summary?.successful_predictions ?? 0}/${result.summary?.total_molecules ?? 0} molecules`,
      });
    } catch (err) {
      setError(err.message);
      toast({
        title: "Batch Prediction Failed",
        description: err.message,
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const exportToCSV = () => {
    if (!results) return;
    
    const headers = ["Name", "SMILES", "Prediction", "Confidence", "BBB+ Probability", "BBB- Probability", "Status"];
    const csvContent = [
      headers.join(","),
      ...results.predictions.map(result => [
        `"${result.name}"`,
        `"${result.smiles}"`,
        result.status === "Success" ? result.prediction : "Failed",
        result.status === "Success" ? result.confidence.toFixed(4) : "",
        result.status === "Success" ? result.probability_bbb_positive.toFixed(4) : "",
        result.status === "Success" ? result.probability_bbb_negative.toFixed(4) : "",
        result.status
      ].join(","))
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `bbb_batch_predictions_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Export Complete",
      description: "Results exported to CSV file",
    });
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 0.8) return { level: "High", color: "bg-green-500" };
    if (confidence >= 0.6) return { level: "Medium", color: "bg-yellow-500" };
    return { level: "Low", color: "bg-red-500" };
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Batch Prediction</CardTitle>
          <CardDescription>
            Enter multiple molecules for batch processing. Format: SMILES,Name (one per line)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="molecules-input">Molecules Data *</Label>
            <Textarea
              id="molecules-input"
              placeholder="CCO,Ethanol&#10;CC(C)O,Isopropanol&#10;CCCCO&#10;..."
              value={moleculesText}
              onChange={(e) => setMoleculesText(e.target.value)}
              className="font-mono text-sm"
              rows={8}
              data-testid="input-molecules"
            />
            <p className="text-xs text-muted-foreground">
              Enter one molecule per line. Format: SMILES,Name or just SMILES
            </p>
          </div>


          <Button 
            onClick={handleBatchPredict} 
            disabled={isLoading || !moleculesText.trim()}
            className="w-full"
            data-testid="button-batch-predict"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing Batch...
              </>
            ) : (
              "Predict Batch"
            )}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {results && (
        <>
          {/* Summary Dashboard */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Batch Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center space-y-1">
                  <div className="text-2xl font-bold" data-testid="text-total">{results.summary?.total_molecules ?? 0}</div>
                  <div className="text-sm text-muted-foreground">Total</div>
                </div>
                <div className="text-center space-y-1">
                  <div className="text-2xl font-bold text-green-600" data-testid="text-successful">{results.summary?.successful_predictions ?? 0}</div>
                  <div className="text-sm text-muted-foreground">Successful</div>
                </div>
                <div className="text-center space-y-1">
                  <div className="text-2xl font-bold text-blue-600" data-testid="text-bbb-positive">{results.summary?.bbb_positive ?? 0}</div>
                  <div className="text-sm text-muted-foreground">BBB+</div>
                </div>
                <div className="text-center space-y-1">
                  <div className="text-2xl font-bold text-red-600" data-testid="text-bbb-negative">{results.summary?.bbb_negative ?? 0}</div>
                  <div className="text-sm text-muted-foreground">BBB-</div>
                </div>
              </div>

              <div className="mt-6 space-y-2">
                <Label>Confidence Distribution</Label>
                <div className="flex gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500 rounded"></div>
                    <span>High: {results.summary?.high_confidence ?? 0}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-yellow-500 rounded"></div>
                    <span>Medium: {results.summary?.medium_confidence ?? 0}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded"></div>
                    <span>Low: {results.summary?.low_confidence ?? 0}</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Results Table */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Detailed Results</CardTitle>
                <Button 
                  onClick={exportToCSV}
                  variant="outline"
                  size="sm"
                  data-testid="button-export-csv"
                >
                  <Download className="mr-2 h-4 w-4" />
                  Export CSV
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Name</th>
                      <th className="text-left p-2">SMILES</th>
                      <th className="text-left p-2">Prediction</th>
                      <th className="text-left p-2">Confidence</th>
                      <th className="text-left p-2">BBB+ Prob</th>
                      <th className="text-left p-2">Processing</th>
                      <th className="text-left p-2">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.predictions?.map((result, index) => (
                      <tr key={index} className="border-b hover:bg-muted/50" data-testid={`row-result-${index}`}>
                        <td className="p-2">{result.name}</td>
                        <td className="p-2 font-mono text-xs">{result.smiles}</td>
                        <td className="p-2">
                          {result.status === "Success" ? (
                            <Badge 
                              variant={result.prediction === "BBB+" ? "default" : "destructive"}
                              className="text-xs"
                            >
                              {result.prediction}
                            </Badge>
                          ) : (
                            <Badge variant="outline" className="text-xs">Failed</Badge>
                          )}
                        </td>
                        <td className="p-2">
                          {result.status === "Success" && (
                            <div className="space-y-1">
                              <Badge variant="outline" className="text-xs">
                                {getConfidenceLevel(result.confidence).level}
                              </Badge>
                              <div className="text-xs text-muted-foreground">
                                {(result.confidence * 100).toFixed(1)}%
                              </div>
                            </div>
                          )}
                        </td>
                        <td className="p-2 font-mono text-xs">
                          {result.status === "Success" && `${(result.probability_bbb_positive * 100).toFixed(2)}%`}
                        </td>
                        <td className="p-2 text-xs">
                          <div>{result.curation_status || "N/A"}</div>
                          <div className="text-muted-foreground">
                            3D: {result.generation_status || "N/A"}
                          </div>
                        </td>
                        <td className="p-2">
                          {result.status === "Success" ? (
                            <Badge variant="outline" className="text-xs text-green-600">Success</Badge>
                          ) : (
                            <Badge variant="destructive" className="text-xs">{result.status}</Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}