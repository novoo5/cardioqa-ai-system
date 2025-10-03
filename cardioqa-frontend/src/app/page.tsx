'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Alert, AlertDescription } from '@/components/ui/alert'
import axios from 'axios'

interface QueryResponse {
  response: string
  safety_score: number
  confidence: string
  knowledge_sources: number
  top_similarity: number
  warnings: string[]
  response_time: number
}

export default function CardioQA() {
  const [query, setQuery] = useState('')
  const [response, setResponse] = useState<QueryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleQuery = async () => {
    if (query.trim().length < 5) {
      setError('Please enter a more specific cardiac health question')
      return
    }

    setLoading(true)
    setError('')
    setResponse(null)

    try {
      const result = await axios.post('http://127.0.0.1:8000/query', {
        query: query.trim(),
        include_metadata: true
      })

      setResponse(result.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get response from CardioQA')
    } finally {
      setLoading(false)
    }
  }

  const getSafetyColor = (score: number) => {
    if (score >= 90) return 'bg-emerald-100 text-emerald-800 border-emerald-200'
    if (score >= 70) return 'bg-amber-100 text-amber-800 border-amber-200'
    return 'bg-red-100 text-red-800 border-red-200'
  }

  const getConfidenceColor = (confidence: string) => {
    if (confidence === 'High') return 'bg-blue-100 text-blue-800 border-blue-200'
    if (confidence === 'Medium') return 'bg-purple-100 text-purple-800 border-purple-200'
    return 'bg-orange-100 text-orange-800 border-orange-200'
  }

  // FIXED: Markdown formatting function
  const formatResponse = (text: string) => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br/>')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Navigation Header */}
      <div className="bg-white shadow-sm border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-red-500 to-pink-500 rounded-lg flex items-center justify-center">
                <span className="text-white text-xl font-bold">🫀</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">CardioQA</h1>
                <p className="text-sm text-slate-600">AI Medical Assistant</p>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-4 text-sm text-slate-600">
              <span className="flex items-center space-x-1">
                <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                <span>API Connected</span>
              </span>
              <span>364 Medical Sources</span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Medical Disclaimer Banner */}
        <div className="mb-8">
          <Alert className="border-amber-200 bg-amber-50/80 backdrop-blur-sm">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-6 h-6 bg-amber-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
                ⚠
              </div>
              <AlertDescription className="text-amber-900 font-medium">
                <strong>Medical Disclaimer:</strong> This AI provides educational information only. 
                Always consult qualified healthcare professionals for medical advice. 
                For emergencies, call emergency services immediately.
              </AlertDescription>
            </div>
          </Alert>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Query Section */}
          <div className="lg:col-span-2 space-y-6">
            {/* Query Card */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-t-lg">
                <CardTitle className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
                    <span className="text-lg">🩺</span>
                  </div>
                  <span>Ask Your Cardiac Health Question</span>
                </CardTitle>
                <CardDescription className="text-blue-100">
                  Get evidence-based information from 364+ medical sources
                </CardDescription>
              </CardHeader>
              <CardContent className="p-6 space-y-4">
                <Textarea
                  placeholder="Example: What are the early warning signs of a heart attack? How can I prevent cardiovascular disease?"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={4}
                  className="resize-none text-slate-700 bg-slate-50 border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent text-base"
                />
                
                <div className="flex space-x-3">
                  <Button 
                    onClick={handleQuery} 
                    disabled={loading}
                    className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium py-3 px-6 rounded-lg transition-all duration-200 shadow-lg"
                  >
                    {loading ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-3"></div>
                        Analyzing Medical Data...
                      </>
                    ) : (
                      <>
                        <span className="mr-2">🔍</span>
                        Get Medical Information
                      </>
                    )}
                  </Button>
                  
                  <Button 
                    variant="outline" 
                    onClick={() => {
                      setQuery('')
                      setResponse(null)
                      setError('')
                    }}
                    className="px-6 border-slate-300 text-slate-700 hover:bg-slate-50"
                  >
                    Clear
                  </Button>
                </div>

                {error && (
                  <Alert className="border-red-200 bg-red-50">
                    <div className="flex items-center space-x-3">
                      <span className="text-red-500 text-lg">❌</span>
                      <AlertDescription className="text-red-800 font-medium">
                        {error}
                      </AlertDescription>
                    </div>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {/* Response Card */}
            {response && (
              <Card className="shadow-lg border-0 bg-white/90 backdrop-blur-sm">
                <CardHeader className="bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-t-lg">
                  <CardTitle className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
                      <span className="text-lg">✅</span>
                    </div>
                    <span>Medical Information Response</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-6">
                  {/* FIXED: Response Text with Proper Markdown */}
                  <div className="mb-6">
                    <div className="prose prose-slate max-w-none text-slate-800 leading-relaxed text-base">
                      <div 
                        className="whitespace-pre-wrap bg-slate-50 p-5 rounded-lg border border-slate-200"
                        dangerouslySetInnerHTML={{
                          __html: formatResponse(response.response)
                        }}
                      />
                    </div>
                  </div>

                  {/* Safety Warnings */}
                  {response.warnings.length > 0 && (
                    <Alert className="border-orange-200 bg-orange-50 mb-6">
                      <div className="flex items-start space-x-3">
                        <span className="text-orange-500 text-lg flex-shrink-0">⚠️</span>
                        <div>
                          <AlertDescription className="text-orange-900 font-medium">
                            <strong>Safety Alerts:</strong>
                            <ul className="list-disc list-inside mt-2 space-y-1">
                              {response.warnings.map((warning, index) => (
                                <li key={index}>{warning}</li>
                              ))}
                            </ul>
                          </AlertDescription>
                        </div>
                      </div>
                    </Alert>
                  )}

                  {/* FIXED: Response Metadata with Dynamic Colors */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className={`p-4 rounded-lg border-2 text-center ${getSafetyColor(response.safety_score)}`}>
                      <div className="text-2xl font-bold mb-1">
                        {response.safety_score}
                      </div>
                      <div className="text-sm font-medium opacity-80">Safety Score</div>
                    </div>
                    
                    <div className={`p-4 rounded-lg border-2 text-center ${getConfidenceColor(response.confidence)}`}>
                      <div className="text-2xl font-bold mb-1">
                        {response.confidence}
                      </div>
                      <div className="text-sm font-medium opacity-80">Confidence</div>
                    </div>
                    
                    <div className="p-4 rounded-lg border-2 bg-indigo-100 text-indigo-800 border-indigo-200 text-center">
                      <div className="text-2xl font-bold mb-1">
                        {response.knowledge_sources}
                      </div>
                      <div className="text-sm font-medium opacity-80">Sources</div>
                    </div>
                    
                    <div className="p-4 rounded-lg border-2 bg-slate-100 text-slate-800 border-slate-200 text-center">
                      <div className="text-2xl font-bold mb-1">
                        {response.response_time}s
                      </div>
                      <div className="text-sm font-medium opacity-80">Response Time</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* System Status */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-slate-700 to-slate-800 text-white rounded-t-lg">
                <CardTitle className="flex items-center space-x-2 text-lg">
                  <span>📊</span>
                  <span>System Status</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4 space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-600">API Status</span>
                  <span className="flex items-center space-x-1">
                    <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                    <span className="text-green-700 font-medium">Online</span>
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-600">Medical Database</span>
                  <span className="text-slate-900 font-medium">364 Q&As</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-600">AI Model</span>
                  <span className="text-slate-900 font-medium">Gemini 2.0</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-600">Specialty</span>
                  <span className="text-slate-900 font-medium">Cardiology</span>
                </div>
              </CardContent>
            </Card>

            {/* Sample Questions */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-purple-600 to-violet-600 text-white rounded-t-lg">
                <CardTitle className="flex items-center space-x-2 text-lg">
                  <span>💡</span>
                  <span>Example Questions</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4 space-y-2">
                {[
                  "What are the warning signs of a heart attack?",
                  "How can I prevent cardiovascular disease?",
                  "What should I do if I have chest pain?",
                  "What are the symptoms of high blood pressure?",
                  "How is stroke prevented and treated?"
                ].map((sample, index) => (
                  <Button 
                    key={index}
                    variant="ghost" 
                    className="w-full justify-start text-left h-auto p-3 text-slate-700 hover:bg-purple-50 hover:text-purple-800 transition-colors duration-200 rounded-lg border border-transparent hover:border-purple-200"
                    onClick={() => setQuery(sample)}
                  >
                    <span className="mr-2 text-purple-500">→</span>
                    <span className="text-sm leading-relaxed">{sample}</span>
                  </Button>
                ))}
              </CardContent>
            </Card>

            {/* Safety Features */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-emerald-600 to-green-600 text-white rounded-t-lg">
                <CardTitle className="flex items-center space-x-2 text-lg">
                  <span>🛡️</span>
                  <span>Safety Features</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4 space-y-3 text-sm">
                <div className="flex items-center space-x-3">
                  <span className="text-green-500">✓</span>
                  <span className="text-slate-700">Medical safety validation</span>
                </div>
                <div className="flex items-center space-x-3">
                  <span className="text-green-500">✓</span>
                  <span className="text-slate-700">Emergency detection</span>
                </div>
                <div className="flex items-center space-x-3">
                  <span className="text-green-500">✓</span>
                  <span className="text-slate-700">Professional disclaimers</span>
                </div>
                <div className="flex items-center space-x-3">
                  <span className="text-green-500">✓</span>
                  <span className="text-slate-700">Evidence-based responses</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
