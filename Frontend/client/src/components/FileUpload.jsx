import { useState, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { AlertCircle, Upload, FileText, Download, Loader2, X } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { apiClient } from "@/lib/apiClient";

export function FileUpload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [smilesColumn, setSmilesColumn] = useState("");
  const [nameColumn, setNameColumn] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const { toast } = useToast();

  const allowedTypes = ['.csv', '.sdf', '.mol'];
  const maxFileSize = 100 * 1024 * 1024; // 100MB

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    const file = files[0];
    
    if (file) {
      validateAndSetFile(file);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      validateAndSetFile(file);
    }
  };

  const validateAndSetFile = (file) => {
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
      toast({
        title: "Invalid File Type",
        description: `Please select a ${allowedTypes.join(', ')} file`,
        variant: "destructive"
      });
      return;
    }

    if (file.size > maxFileSize) {
      toast({
        title: "File Too Large",
        description: "File size must be less than 100MB",
        variant: "destructive"
      });
      return;
    }

    setSelectedFile(file);
    setError(null);
    toast({
      title: "File Selected",
      description: `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`,
    });
  };

  const clearFile = () => {
    setSelectedFile(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      toast({
        title: "Error",
        description: "Please select a file",
        variant: "destructive"
      });
      return;
    }

    if (selectedFile.name.endsWith('.csv') && !smilesColumn) {
      toast({
        title: "Error",
        description: "Please specify the SMILES column for CSV files",
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      if (smilesColumn) formData.append('smiles_column', smilesColumn);
      if (nameColumn) formData.append('name_column', nameColumn);
      
      const result = await apiClient.predictFile(formData, 0.5228, smilesColumn, nameColumn);
      
      // API returns { success: true, data: { predictions: [...], summary: {...} } }
      // Extract the data and normalize the response format
      const normalizedResult = {
        predictions: (result?.data?.predictions || []).map(pred => ({
          ...pred,
          success: pred.status === "Success", // Convert status string to boolean
          error: pred.status !== "Success" ? pred.status : undefined
        })),
        summary: {
          total: result?.data?.summary?.total_molecules || 0,
          successful: result?.data?.summary?.successful_predictions || 0,
          failed: result?.data?.summary?.failed_predictions || 0,
          bbb_positive: result?.data?.summary?.bbb_positive || 0,
          bbb_negative: result?.data?.summary?.bbb_negative || 0,
          file_name: selectedFile.name,
          file_size: selectedFile.size,
          processing_time: 'N/A'
        }
      };
      
      setResults(normalizedResult);
      toast({
        title: "File Processing Complete",
        description: `Processed ${normalizedResult.summary.successful}/${normalizedResult.summary.total} molecules from ${selectedFile.name}`,
      });
    } catch (err) {
      setError(err.message);
      toast({
        title: "File Processing Failed",
        description: err.message,
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const exportToCSV = () => {
    if (!results || !results.predictions) return;
    
    const headers = ["Name", "SMILES", "Prediction", "Confidence", "BBB+ Probability", "BBB- Probability", "Status"];
    const csvContent = [
      headers.join(","),
      ...results.predictions.map(result => [
        `"${result?.name || ''}"`,
        `"${result?.smiles || ''}"`,
        result?.success ? result.prediction : "Failed",
        result?.success ? (result.confidence || 0).toFixed(4) : "",
        result?.success ? (result.probability_bbb_positive || 0).toFixed(4) : "",
        result?.success ? (result.probability_bbb_negative || 0).toFixed(4) : "",
        result?.success ? "Success" : result?.error || "Failed"
      ].join(","))
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `bbb_file_predictions_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Export Complete",
      description: "Results exported to CSV file",
    });
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>File Upload Prediction</CardTitle>
          <CardDescription>
            Upload CSV, SDF, or MOL files for batch molecular analysis
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* File Upload Area */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              isDragging
                ? "border-primary bg-primary/5"
                : selectedFile
                ? "border-green-500 bg-green-50 dark:bg-green-950/20"
                : "border-muted-foreground/25 hover:border-muted-foreground/50"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            data-testid="dropzone-upload"
          >
            {selectedFile ? (
              <div className="space-y-2">
                <FileText className="h-8 w-8 mx-auto text-green-600" />
                <div className="space-y-1">
                  <p className="font-medium">{selectedFile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={clearFile}
                  data-testid="button-clear-file"
                >
                  <X className="h-4 w-4 mr-2" />
                  Remove File
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <Upload className="h-8 w-8 mx-auto text-muted-foreground" />
                <div className="space-y-2">
                  <p className="text-lg font-medium">
                    Drop your file here, or{" "}
                    <button
                      type="button"
                      className="text-primary hover:underline"
                      onClick={() => fileInputRef.current?.click()}
                      data-testid="button-browse-files"
                    >
                      browse
                    </button>
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Supports CSV, SDF, MOL files
                  </p>
                </div>
              </div>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.sdf,.mol"
            onChange={handleFileSelect}
            className="hidden"
            data-testid="input-file"
          />

          {/* CSV Configuration */}
          {selectedFile && selectedFile.name.endsWith('.csv') && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">CSV Configuration</CardTitle>
                <CardDescription>
                  Specify which columns contain the molecular data
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="smiles-column">SMILES Column *</Label>
                    <Input
                      id="smiles-column"
                      placeholder="e.g., SMILES, smiles, Canonical_SMILES"
                      value={smilesColumn}
                      onChange={(e) => setSmilesColumn(e.target.value)}
                      data-testid="input-smiles-column"
                    />
                  </div>
                  
                </div>
              </CardContent>
            </Card>
          )}

          <Button
            onClick={handleUpload}
            disabled={isLoading || !selectedFile}
            className="w-full"
            data-testid="button-upload-process"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing File...
              </>
            ) : (
              <>
                <Upload className="mr-2 h-4 w-4" />
                Process File
              </>
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
          {/* Processing Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Processing Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center space-y-1">
                  <div className="text-2xl font-bold" data-testid="text-file-total">{results?.summary?.total || results?.summary?.total_molecules || 0}</div>
                  <div className="text-sm text-muted-foreground">Total Molecules</div>
                </div>
                <div className="text-center space-y-1">
                  <div className="text-2xl font-bold text-green-600" data-testid="text-file-successful">{results?.summary?.successful || 0}</div>
                  <div className="text-sm text-muted-foreground">Successful</div>
                </div>
                <div className="text-center space-y-1">
                  <div className="text-2xl font-bold text-blue-600" data-testid="text-file-bbb-positive">{results?.summary?.bbb_positive || results?.summary?.bbb_pos || 0}</div>
                  <div className="text-sm text-muted-foreground">BBB+</div>
                </div>
                <div className="text-center space-y-1">
                  <div className="text-2xl font-bold text-red-600" data-testid="text-file-bbb-negative">{results?.summary?.bbb_negative || results?.summary?.bbb_neg || 0}</div>
                  <div className="text-sm text-muted-foreground">BBB-</div>
                </div>
              </div>

              <div className="mt-4 p-4 bg-muted rounded-lg space-y-2">
                <h4 className="font-medium">File Information</h4>
                <div className="text-sm space-y-1">
                  <div><strong>File:</strong> {results?.summary?.file_name || results?.summary?.filename || selectedFile?.name || 'Unknown'}</div>
                  <div><strong>Size:</strong> {results?.summary?.file_size ? (results.summary.file_size / 1024 / 1024).toFixed(2) : selectedFile ? (selectedFile.size / 1024 / 1024).toFixed(2) : '0.00'} MB</div>
                  <div><strong>Processing Time:</strong> {results?.summary?.processing_time || results?.summary?.duration || 'Unknown'}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Export and Results Preview */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Results Preview</CardTitle>
                <Button 
                  onClick={exportToCSV}
                  variant="outline"
                  size="sm"
                  data-testid="button-export-file-csv"
                >
                  <Download className="mr-2 h-4 w-4" />
                  Export Results
                </Button>
              </div>
              <CardDescription>
                Showing first 10 results. Download full results using the export button.
              </CardDescription>
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
                      <th className="text-left p-2">Processing</th>
                      <th className="text-left p-2">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(results?.predictions || []).slice(0, 10).map((result, index) => (
                      <tr key={index} className="border-b hover:bg-muted/50" data-testid={`row-file-result-${index}`}>
                        <td className="p-2">{result?.name || `Molecule ${index + 1}`}</td>
                        <td className="p-2 font-mono text-xs">{result?.smiles || 'N/A'}</td>
                        <td className="p-2">
                          {result?.success ? (
                            <Badge 
                              variant={result.prediction === "BBB+" ? "default" : "destructive"}
                              className="text-xs"
                            >
                              {result.prediction || 'Unknown'}
                            </Badge>
                          ) : (
                            <Badge variant="outline" className="text-xs">Failed</Badge>
                          )}
                        </td>
                        <td className="p-2">
                          {result?.success && result?.confidence !== undefined && (
                            <div className="text-xs text-muted-foreground">
                              {(result.confidence * 100).toFixed(1)}%
                            </div>
                          )}
                        </td>
                        <td className="p-2 text-xs">
                          <div>{result?.curation_status || "N/A"}</div>
                          <div className="text-muted-foreground">
                            3D: {result?.generation_status || "N/A"}
                          </div>
                        </td>
                        <td className="p-2">
                          {result?.success ? (
                            <Badge variant="outline" className="text-xs text-green-600">Success</Badge>
                          ) : (
                            <Badge variant="destructive" className="text-xs">{result?.error || "Failed"}</Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              {(results?.predictions?.length || 0) > 10 && (
                <div className="mt-4 text-center text-sm text-muted-foreground">
                  Showing 10 of {results?.predictions?.length || 0} results. Export to view all.
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}